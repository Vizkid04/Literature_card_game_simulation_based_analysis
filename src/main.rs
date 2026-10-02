use plotters::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

const TOTAL_CARDS: usize = 48;
const CARDS_PER_HALF_SUIT: usize = 6;
const NUM_HALF_SUITS: usize = 8;
const MAX_TURNS: usize = 1000;

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
    pub matrix: Vec<Vec<Possibility>>,
    pub hand_sizes: Vec<usize>,
    pub suit_interest: Vec<Vec<usize>>,
}

impl BotMemory {
    pub fn new(player_id: usize, num_players: usize) -> Self {
        Self {
            player_id,
            num_players,
            matrix: vec![vec![Possibility::Unknown; TOTAL_CARDS]; num_players],
            hand_sizes: vec![0; num_players],
            suit_interest: vec![vec![0; NUM_HALF_SUITS]; num_players],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        let struct_size = std::mem::size_of::<Self>();
        let matrix_heap = self.matrix.capacity() * std::mem::size_of::<Vec<Possibility>>()
            + self.matrix.iter().map(|v| v.capacity() * std::mem::size_of::<Possibility>()).sum::<usize>();
        let hand_sizes_heap = self.hand_sizes.capacity() * std::mem::size_of::<usize>();
        let suit_interest_heap = self.suit_interest.capacity() * std::mem::size_of::<Vec<usize>>()
            + self.suit_interest.iter().map(|v| v.capacity() * std::mem::size_of::<usize>()).sum::<usize>();

        struct_size + matrix_heap + hand_sizes_heap + suit_interest_heap
    }

    pub fn initialize_hand(&mut self, hand: &HashSet<usize>, hand_sizes: &[usize]) {
        self.hand_sizes = hand_sizes.to_vec();
        for card in 0..TOTAL_CARDS {
            if hand.contains(&card) {
                self.matrix[self.player_id][card] = Possibility::Has;
                for p in 0..self.num_players {
                    if p != self.player_id {
                        self.matrix[p][card] = Possibility::DoesNotHave;
                    }
                }
            } else {
                self.matrix[self.player_id][card] = Possibility::DoesNotHave;
            }
        }
    }

    pub fn run_deductions_with_active(&mut self, active_cards: &[bool]) -> usize {
        let mut total_ops = 0;
        let mut changed = true;
        let mut passes = 0;

        while changed && passes < 50 {
            changed = false;
            passes += 1;

            for c in 0..TOTAL_CARDS {
                if !active_cards[c] { continue; }
                let possible: Vec<usize> = (0..self.num_players)
                    .filter(|&p| self.matrix[p][c] != Possibility::DoesNotHave)
                    .collect();
                total_ops += self.num_players;

                if possible.len() == 1 {
                    let holder = possible[0];
                    if self.matrix[holder][c] != Possibility::Has {
                        self.matrix[holder][c] = Possibility::Has;
                        changed = true;
                    }
                }
            }

            for p in 0..self.num_players {
                let known_has_count = (0..TOTAL_CARDS)
                    .filter(|&c| active_cards[c] && self.matrix[p][c] == Possibility::Has)
                    .count();
                total_ops += TOTAL_CARDS;

                if known_has_count == self.hand_sizes[p] {
                    for c in 0..TOTAL_CARDS {
                        if active_cards[c] && self.matrix[p][c] == Possibility::Unknown {
                            self.matrix[p][c] = Possibility::DoesNotHave;
                            changed = true;
                        }
                    }
                }
            }
        }
        total_ops
    }

    pub fn record_ask(&mut self, asker: usize, target: usize, card: usize, success: bool, active_cards: &[bool]) -> usize {
        let suit = card / CARDS_PER_HALF_SUIT;
        self.suit_interest[asker][suit] += 1;

        if success {
            self.matrix[asker][card] = Possibility::Has;
            self.matrix[target][card] = Possibility::DoesNotHave;
            self.hand_sizes[asker] += 1;
            if self.hand_sizes[target] > 0 {
                self.hand_sizes[target] -= 1;
            }
        } else {
            self.matrix[asker][card] = Possibility::DoesNotHave;
            self.matrix[target][card] = Possibility::DoesNotHave;
        }

        self.run_deductions_with_active(active_cards)
    }

    pub fn record_claim(&mut self, hs: usize, active_cards: &[bool]) -> usize {
        let start = hs * CARDS_PER_HALF_SUIT;
        for c in start..(start + CARDS_PER_HALF_SUIT) {
            for p in 0..self.num_players {
                self.matrix[p][c] = Possibility::DoesNotHave;
            }
        }
        self.run_deductions_with_active(active_cards)
    }

    pub fn evaluate_best_ask(&self, my_hand: &HashSet<usize>, active_cards: &[bool]) -> Option<(usize, usize, f64)> {
        let my_team = self.player_id % 2;
        let mut best_move = None;
        let mut max_score = -1.0;

        let my_half_suits: HashSet<usize> = my_hand.iter().map(|&c| c / CARDS_PER_HALF_SUIT).collect();

        for &hs in &my_half_suits {
            let start = hs * CARDS_PER_HALF_SUIT;
            for c in start..(start + CARDS_PER_HALF_SUIT) {
                if my_hand.contains(&c) || !active_cards[c] { continue; }

                for target in 0..self.num_players {
                    if target % 2 == my_team || self.hand_sizes[target] == 0 { continue; }

                    let status = self.matrix[target][c];
                    let score = match status {
                        Possibility::Has => 100.0,
                        Possibility::DoesNotHave => -100.0,
                        Possibility::Unknown => 1.0 + (self.suit_interest[target][hs] as f64) * 2.0,
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

    pub fn evaluate_claim_risk(&self, hs: usize, my_hand: &HashSet<usize>, active_cards: &[bool]) -> Option<Vec<(usize, usize)>> {
        let start = hs * CARDS_PER_HALF_SUIT;
        let active_hs_cards: Vec<usize> = (start..(start + CARDS_PER_HALF_SUIT)).filter(|&c| active_cards[c]).collect();

        if active_hs_cards.is_empty() || !active_hs_cards.iter().any(|c| my_hand.contains(c)) {
            return None;
        }

        let mut declaration = Vec::new();

        for &c in &active_hs_cards {
            if my_hand.contains(&c) {
                declaration.push((c, self.player_id));
            } else {
                let owner = (0..self.num_players).find(|&p| self.matrix[p][c] == Possibility::Has);
                if let Some(p) = owner {
                    declaration.push((c, p));
                } else {
                    return None;
                }
            }
        }

        if declaration.len() == active_hs_cards.len() { Some(declaration) } else { None }
    }
}

pub struct LiteratureGame {
    pub num_players: usize,
    pub hands: Vec<HashSet<usize>>,
    pub bots: Vec<BotMemory>,
    pub active_cards: Vec<bool>,
    pub half_suit_claimed: Vec<bool>,
    pub team_scores: [usize; 2],
    pub current_player: usize,
    pub turns: usize,
}

impl LiteratureGame {
    pub fn new(num_players: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut deck: Vec<usize> = (0..TOTAL_CARDS).collect();
        deck.shuffle(&mut rng);

        let mut hands = vec![HashSet::new(); num_players];
        for (i, &card) in deck.iter().enumerate() {
            hands[i % num_players].insert(card);
        }

        let hand_sizes: Vec<usize> = hands.iter().map(|h| h.len()).collect();
        let mut bots = vec![BotMemory::new(0, num_players); num_players];

        for i in 0..num_players {
            bots[i] = BotMemory::new(i, num_players);
            bots[i].initialize_hand(&hands[i], &hand_sizes);
        }

        Self {
            num_players,
            hands,
            bots,
            active_cards: vec![true; TOTAL_CARDS],
            half_suit_claimed: vec![false; NUM_HALF_SUITS],
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
            if cand % 2 == team && !self.hands[cand].is_empty() {
                return Some(cand);
            }
        }
        (0..self.num_players).find(|&p| !self.hands[p].is_empty())
    }

    pub fn play_step(&mut self) -> usize {
        let mut ops = 0;
        self.turns += 1;

        let mut p = self.current_player;
        if self.hands[p].is_empty() {
            if let Some(nxt) = self.get_next_player(p) {
                self.current_player = nxt;
                p = nxt;
            } else {
                return ops;
            }
        }

        let my_team = p % 2;
        let opponents_have_cards = (0..self.num_players).any(|o| o % 2 != my_team && !self.hands[o].is_empty());

        if !opponents_have_cards {
            for hs in 0..NUM_HALF_SUITS {
                if !self.half_suit_claimed[hs] {
                    self.half_suit_claimed[hs] = true;
                    self.team_scores[my_team] += 1;
                }
            }
            return ops;
        }

        for hs in 0..NUM_HALF_SUITS {
            if self.half_suit_claimed[hs] { continue; }
            if let Some(claim) = self.bots[p].evaluate_claim_risk(hs, &self.hands[p], &self.active_cards) {
                let is_correct = claim.iter().all(|&(card, owner)| self.hands[owner].contains(&card));
                let claiming_team = p % 2;

                if is_correct {
                    self.team_scores[claiming_team] += 1;
                } else {
                    self.team_scores[1 - claiming_team] += 1;
                }

                self.half_suit_claimed[hs] = true;
                let start = hs * CARDS_PER_HALF_SUIT;
                for c in start..(start + CARDS_PER_HALF_SUIT) {
                    self.active_cards[c] = false;
                    for hand in self.hands.iter_mut() {
                        hand.remove(&c);
                    }
                }

                for bot in self.bots.iter_mut() {
                    ops += bot.record_claim(hs, &self.active_cards);
                }
                return ops;
            }
        }

        if let Some((target, card, _)) = self.bots[p].evaluate_best_ask(&self.hands[p], &self.active_cards) {
            let success = self.hands[target].contains(&card);
            if success {
                self.hands[target].remove(&card);
                self.hands[p].insert(card);
            }

            for bot in self.bots.iter_mut() {
                ops += bot.record_ask(p, target, card, success, &self.active_cards);
            }

            if !success {
                self.current_player = target;
            }
            return ops;
        }

        let my_hand: Vec<usize> = self.hands[p].iter().copied().collect();
        if !my_hand.is_empty() {
            let my_suits: HashSet<usize> = my_hand.iter().map(|&c| c / CARDS_PER_HALF_SUIT).collect();

            for hs in my_suits {
                let start = hs * CARDS_PER_HALF_SUIT;
                let missing: Vec<usize> = (start..(start + CARDS_PER_HALF_SUIT))
                    .filter(|&c| !self.hands[p].contains(&c) && self.active_cards[c])
                    .collect();

                for c in missing {
                    let candidates: Vec<usize> = (0..self.num_players)
                        .filter(|&o| o % 2 != p % 2 && !self.hands[o].is_empty() && self.bots[p].matrix[o][c] != Possibility::DoesNotHave)
                        .collect();

                    if let Some(&target) = candidates.first() {
                        let success = self.hands[target].contains(&c);
                        if success {
                            self.hands[target].remove(&c);
                            self.hands[p].insert(c);
                        }
                        for bot in self.bots.iter_mut() {
                            ops += bot.record_ask(p, target, c, success, &self.active_cards);
                        }
                        if !success {
                            self.current_player = target;
                        }
                        return ops;
                    }
                }
            }

            for hs in (0..NUM_HALF_SUITS).filter(|&hs| !self.half_suit_claimed[hs]) {
                if my_hand.iter().any(|&c| c / CARDS_PER_HALF_SUIT == hs) {
                    let start = hs * CARDS_PER_HALF_SUIT;
                    let hs_cards: Vec<usize> = (start..(start + CARDS_PER_HALF_SUIT)).filter(|&c| self.active_cards[c]).collect();
                    let mut guess_map = Vec::new();

                    for &c in &hs_cards {
                        if self.hands[p].contains(&c) {
                            guess_map.push((c, p));
                        } else {
                            let candidate = (0..self.num_players).find(|&o| !self.hands[o].is_empty() && self.bots[p].matrix[o][c] != Possibility::DoesNotHave)
                                .or_else(|| (0..self.num_players).find(|&o| o % 2 == my_team && !self.hands[o].is_empty()))
                                .or_else(|| (0..self.num_players).find(|&o| !self.hands[o].is_empty()));
                            if let Some(owner) = candidate {
                                guess_map.push((c, owner));
                            }
                        }
                    }

                    if guess_map.len() == hs_cards.len() {
                        let is_correct = guess_map.iter().all(|&(card, owner)| self.hands[owner].contains(&card));
                        if is_correct { self.team_scores[my_team] += 1; } else { self.team_scores[1 - my_team] += 1; }
                        self.half_suit_claimed[hs] = true;
                        for c in start..(start + CARDS_PER_HALF_SUIT) {
                            self.active_cards[c] = false;
                            for hand in self.hands.iter_mut() { hand.remove(&c); }
                        }
                        for bot in self.bots.iter_mut() { ops += bot.record_claim(hs, &self.active_cards); }
                        return ops;
                    }
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
    let max_n = results.last().map(|r| r.n as u32).unwrap_or(16);
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
    let player_counts = vec![4, 6, 8, 10, 12, 14, 16];
    let iterations = 100;

    println!(" ===================================================================================================== ");
    println!("  RUNNING LITERATURE GAME COMPLEXITY BENCHMARK (100 Iterations per Player Count) ");
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
