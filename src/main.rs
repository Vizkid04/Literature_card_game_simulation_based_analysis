use plotters::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::time::Instant;

const TOTAL_CARDS: usize = 48;
const CARDS_PER_HALF_SUIT: usize = 6;
const NUM_HALF_SUITS: usize = 8;
const MAX_TURNS: usize = 1000;
const ALL_CARDS_MASK: u64 = (1u64 << TOTAL_CARDS) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Possibility {
    Unknown,
    Has,
    DoesNotHave,
}

#[derive(Clone)]
pub struct BotMemory {
    pub player_id: usize,
    pub num_players: usize,
    pub has_mask: Vec<u64>,
    pub does_not_have_mask: Vec<u64>,
    pub hand_sizes: Vec<u8>,
    pub suit_interest: Vec<[u8; NUM_HALF_SUITS]>,
    pub asked_suit_mask: Vec<u8>,
}

impl BotMemory {
    pub fn new(player_id: usize, num_players: usize) -> Self {
        Self {
            player_id,
            num_players,
            has_mask: vec![0; num_players],
            does_not_have_mask: vec![0; num_players],
            hand_sizes: vec![0; num_players],
            suit_interest: vec![[0; NUM_HALF_SUITS]; num_players],
            asked_suit_mask: vec![0; num_players],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.has_mask.capacity() * std::mem::size_of::<u64>()
            + self.does_not_have_mask.capacity() * std::mem::size_of::<u64>()
            + self.hand_sizes.capacity() * std::mem::size_of::<u8>()
            + self.suit_interest.capacity() * std::mem::size_of::<[u8; NUM_HALF_SUITS]>()
            + self.asked_suit_mask.capacity() * std::mem::size_of::<u8>()
    }

    pub fn initialize_hand(&mut self, hand: u64, hand_sizes: &[usize]) {
        self.hand_sizes = hand_sizes.iter().map(|&s| s as u8).collect();
        self.has_mask[self.player_id] = hand;
        self.does_not_have_mask[self.player_id] = !hand & ALL_CARDS_MASK;

        for p in 0..self.num_players {
            if p != self.player_id {
                self.does_not_have_mask[p] |= hand;
            }
        }
    }

    pub fn run_deductions_with_active(&mut self, active_cards: u64) -> usize {
        let mut total_ops = 0;
        let mut changed = true;
        let mut passes = 0;

        while changed && passes < 20 {
            changed = false;
            passes += 1;

            // 1. Single Candidate Rule & Cross Exclusion
            for c in 0..TOTAL_CARDS {
                let card_bit = 1u64 << c;
                if (active_cards & card_bit) == 0 {
                    continue;
                }

                let mut possible_count = 0;
                let mut last_possible = 0;

                for p in 0..self.num_players {
                    total_ops += 1;
                    if (self.has_mask[p] & card_bit) != 0 {
                        for o in 0..self.num_players {
                            if o != p && (self.does_not_have_mask[o] & card_bit) == 0 {
                                self.does_not_have_mask[o] |= card_bit;
                                changed = true;
                            }
                        }
                        possible_count = 0;
                        break;
                    }

                    if (self.does_not_have_mask[p] & card_bit) == 0 {
                        possible_count += 1;
                        last_possible = p;
                    }
                }

                if possible_count == 1 && (self.has_mask[last_possible] & card_bit) == 0 {
                    self.has_mask[last_possible] |= card_bit;
                    changed = true;
                }
            }

            // 2. Hand Size Exhaustion Rule
            for p in 0..self.num_players {
                total_ops += 1;
                let known_has = (self.has_mask[p] & active_cards).count_ones() as u8;
                if known_has == self.hand_sizes[p] {
                    let unknown_mask = active_cards & !self.has_mask[p];
                    if (self.does_not_have_mask[p] & unknown_mask) != unknown_mask {
                        self.does_not_have_mask[p] |= unknown_mask;
                        changed = true;
                    }
                }
            }

            // 3. Suit Interest Constraint: Asker must hold at least 1 card in asked suit
            for p in 0..self.num_players {
                for hs in 0..NUM_HALF_SUITS {
                    total_ops += 1;
                    if (self.asked_suit_mask[p] & (1 << hs)) != 0 {
                        let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
                        let active_hs = hs_mask & active_cards;
                        if active_hs == 0 || (self.has_mask[p] & active_hs) != 0 {
                            continue;
                        }

                        let candidates = active_hs & !self.does_not_have_mask[p];
                        if candidates.count_ones() == 1 && (self.has_mask[p] & candidates) == 0 {
                            self.has_mask[p] |= candidates;
                            changed = true;
                        }
                    }
                }
            }
        }
        total_ops
    }

    pub fn record_ask(&mut self, asker: usize, target: usize, card: usize, success: bool, active_cards: u64) -> usize {
        let suit = card / CARDS_PER_HALF_SUIT;
        let card_bit = 1u64 << card;

        self.suit_interest[asker][suit] = self.suit_interest[asker][suit].saturating_add(1);
        self.asked_suit_mask[asker] |= 1 << suit;
        self.does_not_have_mask[asker] |= card_bit;

        if success {
            self.has_mask[asker] |= card_bit;
            self.does_not_have_mask[asker] &= !card_bit;
            self.does_not_have_mask[target] |= card_bit;
            self.has_mask[target] &= !card_bit;

            self.hand_sizes[asker] += 1;
            if self.hand_sizes[target] > 0 {
                self.hand_sizes[target] -= 1;
            }
        } else {
            self.does_not_have_mask[target] |= card_bit;
        }

        self.run_deductions_with_active(active_cards)
    }

    pub fn record_claim(&mut self, hs: usize, active_cards: u64) -> usize {
        let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
        for p in 0..self.num_players {
            self.has_mask[p] &= !hs_mask;
            self.does_not_have_mask[p] |= hs_mask;
        }
        self.run_deductions_with_active(active_cards)
    }

    pub fn evaluate_best_ask(&self, my_hand: u64, active_cards: u64) -> Option<(usize, usize, f64)> {
        let my_team = self.player_id % 2;
        let mut best_move = None;
        let mut max_score = -1.0;

        for hs in 0..NUM_HALF_SUITS {
            let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
            if (my_hand & hs_mask) == 0 {
                continue;
            }

            let start = hs * CARDS_PER_HALF_SUIT;
            for c_offset in 0..CARDS_PER_HALF_SUIT {
                let c = start + c_offset;
                let card_bit = 1u64 << c;

                if (my_hand & card_bit) != 0 || (active_cards & card_bit) == 0 {
                    continue;
                }

                for target in 0..self.num_players {
                    if target % 2 == my_team || self.hand_sizes[target] == 0 {
                        continue;
                    }

                    let known_has = (self.has_mask[target] & card_bit) != 0;
                    let known_not_has = (self.does_not_have_mask[target] & card_bit) != 0;

                    let score = if known_has {
                        1000.0
                    } else if known_not_has {
                        -1000.0
                    } else {
                        1.0 + (self.suit_interest[target][hs] as f64) * 5.0 + (self.hand_sizes[target] as f64) * 0.5
                    };

                    if score > max_score && score > 0.0 {
                        max_score = score;
                        best_move = Some((target, c, score));
                    }
                }
            }
        }
        best_move
    }

    pub fn evaluate_claim_risk(&self, hs: usize, my_hand: u64, active_cards: u64) -> Option<Vec<(usize, usize)>> {
        let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
        let active_hs = hs_mask & active_cards;

        if active_hs == 0 || (my_hand & active_hs) == 0 {
            return None;
        }

        let start = hs * CARDS_PER_HALF_SUIT;
        let mut declaration = Vec::with_capacity(CARDS_PER_HALF_SUIT);

        for c_offset in 0..CARDS_PER_HALF_SUIT {
            let c = start + c_offset;
            let card_bit = 1u64 << c;

            if (active_cards & card_bit) == 0 {
                continue;
            }

            if (my_hand & card_bit) != 0 {
                declaration.push((c, self.player_id));
            } else {
                let owner = (0..self.num_players).find(|&p| (self.has_mask[p] & card_bit) != 0);
                if let Some(p) = owner {
                    declaration.push((c, p));
                } else {
                    return None;
                }
            }
        }

        if declaration.len() == active_hs.count_ones() as usize {
            Some(declaration)
        } else {
            None
        }
    }
}

pub struct LiteratureGame {
    pub num_players: usize,
    pub hands: Vec<u64>,
    pub bots: Vec<BotMemory>,
    pub active_cards: u64,
    pub half_suit_claimed: u8,
    pub team_scores: [usize; 2],
    pub current_player: usize,
    pub turns: usize,
}

impl LiteratureGame {
    pub fn new(num_players: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut deck: Vec<usize> = (0..TOTAL_CARDS).collect();
        deck.shuffle(&mut rng);

        let mut hands = vec![0u64; num_players];
        for (i, &card) in deck.iter().enumerate() {
            hands[i % num_players] |= 1u64 << card;
        }

        let hand_sizes: Vec<usize> = hands.iter().map(|h| h.count_ones() as usize).collect();
        let mut bots = Vec::with_capacity(num_players);

        for i in 0..num_players {
            let mut bot = BotMemory::new(i, num_players);
            bot.initialize_hand(hands[i], &hand_sizes);
            bots.push(bot);
        }

        Self {
            num_players,
            hands,
            bots,
            active_cards: ALL_CARDS_MASK,
            half_suit_claimed: 0,
            team_scores: [0, 0],
            current_player: 0,
            turns: 0,
        }
    }

    pub fn is_over(&self) -> bool {
        self.team_scores[0] + self.team_scores[1] == NUM_HALF_SUITS || self.turns >= MAX_TURNS
    }

    pub fn get_next_player(&self, start: usize) -> Option<usize> {
        let team = start % 2;
        for i in 0..self.num_players {
            let cand = (start + i) % self.num_players;
            if cand % 2 == team && self.hands[cand] != 0 {
                return Some(cand);
            }
        }
        (0..self.num_players).find(|&p| self.hands[p] != 0)
    }

    pub fn play_step(&mut self) -> usize {
        let mut ops = 0;
        self.turns += 1;

        let mut p = self.current_player;
        if self.hands[p] == 0 {
            if let Some(nxt) = self.get_next_player(p) {
                self.current_player = nxt;
                p = nxt;
            } else {
                return ops;
            }
        }

        let my_team = p % 2;
        let opponents_have_cards = (0..self.num_players).any(|o| o % 2 != my_team && self.hands[o] != 0);

        if !opponents_have_cards {
            for hs in 0..NUM_HALF_SUITS {
                if (self.half_suit_claimed & (1 << hs)) == 0 {
                    self.half_suit_claimed |= 1 << hs;
                    self.team_scores[my_team] += 1;
                }
            }
            return ops;
        }

        // 1. Evaluate safe claims
        for hs in 0..NUM_HALF_SUITS {
            if (self.half_suit_claimed & (1 << hs)) != 0 {
                continue;
            }
            if let Some(claim) = self.bots[p].evaluate_claim_risk(hs, self.hands[p], self.active_cards) {
                let is_correct = claim.iter().all(|&(card, owner)| (self.hands[owner] & (1u64 << card)) != 0);
                let claiming_team = p % 2;

                if is_correct {
                    self.team_scores[claiming_team] += 1;
                } else {
                    self.team_scores[1 - claiming_team] += 1;
                }

                self.half_suit_claimed |= 1 << hs;
                let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
                self.active_cards &= !hs_mask;

                for hand in self.hands.iter_mut() {
                    *hand &= !hs_mask;
                }

                for bot in self.bots.iter_mut() {
                    ops += bot.record_claim(hs, self.active_cards);
                }
                return ops;
            }
        }

        // 2. Evaluate optimal asking strategy
        if let Some((target, card, _)) = self.bots[p].evaluate_best_ask(self.hands[p], self.active_cards) {
            let card_bit = 1u64 << card;
            let success = (self.hands[target] & card_bit) != 0;

            if success {
                self.hands[target] &= !card_bit;
                self.hands[p] |= card_bit;
            }

            for bot in self.bots.iter_mut() {
                ops += bot.record_ask(p, target, card, success, self.active_cards);
            }

            if !success {
                self.current_player = target;
            }
            return ops;
        }

        // 3. Fallback ask
        let my_hand = self.hands[p];
        if my_hand != 0 {
            for hs in 0..NUM_HALF_SUITS {
                let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
                if (my_hand & hs_mask) == 0 {
                    continue;
                }

                let start = hs * CARDS_PER_HALF_SUIT;
                for c_offset in 0..CARDS_PER_HALF_SUIT {
                    let c = start + c_offset;
                    let card_bit = 1u64 << c;

                    if (my_hand & card_bit) != 0 || (self.active_cards & card_bit) == 0 {
                        continue;
                    }

                    for o in 0..self.num_players {
                        if o % 2 != my_team && self.hands[o] != 0 && (self.bots[p].does_not_have_mask[o] & card_bit) == 0 {
                            let success = (self.hands[o] & card_bit) != 0;
                            if success {
                                self.hands[o] &= !card_bit;
                                self.hands[p] |= card_bit;
                            }

                            for bot in self.bots.iter_mut() {
                                ops += bot.record_ask(p, o, c, success, self.active_cards);
                            }

                            if !success {
                                self.current_player = o;
                            }
                            return ops;
                        }
                    }
                }
            }

            // 4. Last resort guess claim
            for hs in 0..NUM_HALF_SUITS {
                if (self.half_suit_claimed & (1 << hs)) != 0 {
                    continue;
                }

                let hs_mask = 0x3Fu64 << (hs * CARDS_PER_HALF_SUIT);
                if (my_hand & hs_mask) == 0 {
                    continue;
                }

                let start = hs * CARDS_PER_HALF_SUIT;
                let mut guess_map = Vec::with_capacity(CARDS_PER_HALF_SUIT);

                for c_offset in 0..CARDS_PER_HALF_SUIT {
                    let c = start + c_offset;
                    let card_bit = 1u64 << c;

                    if (self.active_cards & card_bit) == 0 {
                        continue;
                    }

                    if (my_hand & card_bit) != 0 {
                        guess_map.push((c, p));
                    } else {
                        let candidate = (0..self.num_players)
                            .find(|&o| self.hands[o] != 0 && (self.bots[p].does_not_have_mask[o] & card_bit) == 0)
                            .or_else(|| (0..self.num_players).find(|&o| o % 2 == my_team && self.hands[o] != 0))
                            .or_else(|| (0..self.num_players).find(|&o| self.hands[o] != 0));

                        if let Some(owner) = candidate {
                            guess_map.push((c, owner));
                        }
                    }
                }

                let active_hs = hs_mask & self.active_cards;
                if guess_map.len() == active_hs.count_ones() as usize {
                    let is_correct = guess_map.iter().all(|&(card, owner)| (self.hands[owner] & (1u64 << card)) != 0);
                    if is_correct {
                        self.team_scores[my_team] += 1;
                    } else {
                        self.team_scores[1 - my_team] += 1;
                    }

                    self.half_suit_claimed |= 1 << hs;
                    self.active_cards &= !hs_mask;

                    for hand in self.hands.iter_mut() {
                        *hand &= !hs_mask;
                    }

                    for bot in self.bots.iter_mut() {
                        ops += bot.record_claim(hs, self.active_cards);
                    }
                    return ops;
                }
            }
        }

        if let Some(nxt) = self.get_next_player(p + 1) {
            self.current_player = nxt;
        }

        ops
    }
}

pub struct BenchmarkMetrics {
    pub n: usize,
    pub avg_turns: f64,
    pub avg_ops: f64,
    pub ops_per_turn_bot: f64,
    pub avg_time_us: f64,
    pub mem_per_bot: f64,
    pub total_sys_mem: f64,
}

pub fn run_benchmark_for_n(n: usize, iterations: usize) -> BenchmarkMetrics {
    let start_time = Instant::now();

    let results: Vec<(usize, usize)> = (0..iterations)
        .into_par_iter()
        .map(|_| {
            let mut game = LiteratureGame::new(n);
            let mut total_ops = 0;
            while !game.is_over() {
                total_ops += game.play_step();
            }
            (game.turns, total_ops)
        })
        .collect();

    let elapsed = start_time.elapsed();

    let total_turns: usize = results.iter().map(|(t, _)| t).sum();
    let total_ops: usize = results.iter().map(|(_, o)| o).sum();

    let avg_turns = total_turns as f64 / iterations as f64;
    let avg_ops = total_ops as f64 / iterations as f64;
    let ops_per_turn_bot = avg_ops / (avg_turns * n as f64);
    let avg_time_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;

    let sample_bot = BotMemory::new(0, n);
    let mem_per_bot = sample_bot.memory_bytes() as f64;
    let total_sys_mem = mem_per_bot * n as f64;

    BenchmarkMetrics {
        n,
        avg_turns,
        avg_ops,
        ops_per_turn_bot,
        avg_time_us,
        mem_per_bot,
        total_sys_mem,
    }
}

fn render_svg_plot<F>(
    results: &[BenchmarkMetrics],
    filename: &str,
    title: &str,
    y_label: &str,
    extractor: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&BenchmarkMetrics) -> f64,
{
    let root = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_n = results.first().map(|r| r.n as u32).unwrap_or(4);
    let max_n = results.last().map(|r| r.n as u32).unwrap_or(60);
    let max_y = results.iter().map(&extractor).fold(0.0f64, f64::max) * 1.15;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(min_n..max_n + 1, 0.0..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Number of Players (N)")
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 14))
        .draw()?;

    let points: Vec<(u32, f64)> = results.iter().map(|r| (r.n as u32, extractor(r))).collect();

    chart.draw_series(LineSeries::new(points.clone(), BLUE.stroke_width(3)))?;

    chart.draw_series(points.into_iter().map(|(x, y)| {
        Circle::new((x, y), 5, BLUE.filled())
    }))?;

    root.present()?;
    println!("Generated SVG vector chart: {}", filename);
    Ok(())
}

fn generate_vector_graphs(results: &[BenchmarkMetrics]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nGenerating SVG Vector Plot Files...");

    render_svg_plot(
        results,
        "1_avg_turns_vs_n.svg",
        "Average Turns per Game vs N",
        "Avg Turns",
        |r| r.avg_turns,
    )?;

    render_svg_plot(
        results,
        "2_avg_ops_vs_n.svg",
        "Average Operations per Game vs N",
        "Avg Operations",
        |r| r.avg_ops,
    )?;

    render_svg_plot(
        results,
        "3_ops_per_turn_bot_vs_n.svg",
        "Operations / Turn / Bot vs N",
        "Ops / Turn / Bot",
        |r| r.ops_per_turn_bot,
    )?;

    render_svg_plot(
        results,
        "4_avg_time_vs_n.svg",
        "Average Execution Time vs N",
        "Avg Time (microseconds)",
        |r| r.avg_time_us,
    )?;

    render_svg_plot(
        results,
        "5_mem_per_bot_vs_n.svg",
        "Memory per Bot vs N",
        "Memory / Bot (Bytes)",
        |r| r.mem_per_bot,
    )?;

    render_svg_plot(
        results,
        "6_total_sys_mem_vs_n.svg",
        "Total System Memory vs N",
        "Total System Memory (Bytes)",
        |r| r.total_sys_mem,
    )?;

    Ok(())
}

fn main() {
    let player_counts = vec![4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60];
    let iterations = 100;

    println!(" ===================================================================================================== ");
    println!("   RUNNING LITERATURE GAME COMPLEXITY BENCHMARK (100 Iterations per Player Count) ");
    println!(" ===================================================================================================== \n");

    let table_border = " +-----+------------+-------------------+--------------------+------------------+-------------------+--------------------+ ";
    let table_header = " |  N  | Avg Turns  | Avg Ops / Game    | Ops / Turn / Bot   | Avg Time (us)    | Mem / Bot (Bytes) | Total System Mem   | ";

    println!("{}", table_border);
    println!("{}", table_header);
    println!("{}", table_border);

    let mut metrics_history = Vec::new();

    for &n in &player_counts {
        let metrics = run_benchmark_for_n(n, iterations);
        println!(
            " | {:^3} | {:>10.1} | {:>17.0} | {:>18.1} | {:>16.1} | {:>17.1} | {:>18.1} | ",
            metrics.n,
            metrics.avg_turns,
            metrics.avg_ops,
            metrics.ops_per_turn_bot,
            metrics.avg_time_us,
            metrics.mem_per_bot,
            metrics.total_sys_mem
        );
        metrics_history.push(metrics);
    }

    println!("{}", table_border);

    if let Err(e) = generate_vector_graphs(&metrics_history) {
        eprintln!("Failed to generate SVG charts: {}", e);
    }
}
