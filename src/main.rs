use plotters::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, BufWriter, Write};

const CARDS: usize = 48;
const BOOKS: usize = 8;
const BOOK_SIZE: usize = 6;
const ITERATIONS: usize = 100;
const MAX_MOVES: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
	Hit,
	Miss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ask {
	target: usize,
	card: usize,
}

#[derive(Clone)]
struct Game {
	active: usize,
	team: Vec<u8>,
	hand: Vec<Vec<usize>>,
	owner: [usize; CARDS],
	resolved: [bool; BOOKS],
	turn: usize,
	moves: usize,
	score: [usize; 2],
}

#[derive(Clone)]
struct Belief {
	possible: [u64; CARDS],
	known_misses: Vec<[bool; CARDS]>,
}

#[derive(Clone, Copy, Default)]
struct ResultRow {
	n: usize,
	active: usize,
	processing_game: f64,
	processing_game_sd: f64,
	processing_game_per_player: f64,
	processing_game_per_player_sd: f64,
	processing_turn: f64,
	processing_turn_sd: f64,
	memory_player: f64,
	memory_player_sd: f64,
	moves: f64,
	moves_sd: f64,
}

fn log2(x: f64) -> f64 {
	if x <= 0.0 {
		0.0
	} else {
		x.log2()
	}
}

fn book(card: usize) -> usize {
	card / BOOK_SIZE
}

fn active_players(n: usize) -> usize {
	n.min(CARDS)
}

fn make_team(active: usize) -> Vec<u8> {
	(0..active).map(|p| (p % 2) as u8).collect()
}

fn first_nonempty(hand: &[Vec<usize>]) -> Option<usize> {
	hand.iter().position(|cards| !cards.is_empty())
}

fn next_nonempty(game: &Game, start: usize) -> Option<usize> {
	if game.active == 0 {
		return None;
	}

	for offset in 1..=game.active {
		let candidate = (start + offset) % game.active;

		if !game.hand[candidate].is_empty() {
			return Some(candidate);
		}
	}

	None
}

fn setup(n: usize, rng: &mut impl rand::Rng) -> Game {
	let active = active_players(n);

	let mut deck: Vec<usize> = (0..CARDS).collect();
	deck.shuffle(rng);

	let mut hand = vec![Vec::new(); active];
	let mut owner = [0usize; CARDS];

	for (i, card) in deck.into_iter().enumerate() {
		let player = i % active;
		hand[player].push(card);
		owner[card] = player;
	}

	let turn = first_nonempty(&hand).expect("deck is non-empty");

	Game {
		active,
		team: make_team(active),
		hand,
		owner,
		resolved: [false; BOOKS],
		turn,
		moves: 0,
		score: [0, 0],
	}
}

fn initial_belief(game: &Game, me: usize) -> Belief {
	let mut possible = [0u64; CARDS];
	let mut possible_players = 0u64;

	for player in 0..game.active {
		if !game.hand[player].is_empty() {
			possible_players |= 1u64 << player;
		}
	}

	for card in 0..CARDS {
		if game.hand[me].contains(&card) {
			possible[card] = 1u64 << me;
		} else {
			possible[card] =
				possible_players & !(1u64 << me);
		}
	}

	Belief {
		possible,
		known_misses: vec![[false; CARDS]; game.active],
	}
}

fn eliminate_player(belief: &mut Belief, player: usize) {
	let mask = !(1u64 << player);

	for card in 0..CARDS {
		belief.possible[card] &= mask;
	}
}

fn synchronize_own_hand(
	game: &Game,
	belief: &mut Belief,
	me: usize,
) {
	for card in 0..CARDS {
		if game.hand[me].contains(&card) {
			belief.possible[card] = 1u64 << me;
		}
	}
}

fn update_belief_after_question(
	game: &Game,
	belief: &mut Belief,
	ask: Ask,
	outcome: Outcome,
	asker: usize,
	observer: usize,
) {
	match outcome {
		Outcome::Hit => {
			belief.possible[ask.card] = 1u64 << asker;
		}
		Outcome::Miss => {
			belief.possible[ask.card] &=
				!(1u64 << ask.target);

			belief.known_misses[ask.target][ask.card] =
				true;
		}
	}

	for player in 0..game.active {
		if game.hand[player].is_empty() {
			eliminate_player(belief, player);
		}
	}

	synchronize_own_hand(game, belief, observer);
}

fn unresolved_card_entropy(
	game: &Game,
	belief: &Belief,
	card: usize,
) -> f64 {
	if game.resolved[book(card)] {
		return 0.0;
	}

	let possibilities =
		belief.possible[card].count_ones();

	if possibilities <= 1 {
		0.0
	} else {
		log2(possibilities as f64)
	}
}

fn memory_entropy(
	game: &Game,
	belief: &Belief,
) -> f64 {
	(0..CARDS)
		.map(|card| {
			unresolved_card_entropy(
				game,
				belief,
				card,
			)
		})
		.sum()
}

fn binary_entropy(p: f64) -> f64 {
	if p <= 0.0 || p >= 1.0 {
		return 0.0;
	}

	-p * log2(p) - (1.0 - p) * log2(1.0 - p)
}

fn legal_asks(
	game: &Game,
	player: usize,
	belief: &Belief,
) -> Vec<Ask> {
	if game.hand[player].is_empty() {
		return Vec::new();
	}

	let mut asks = Vec::new();
	let mut represented = [false; BOOKS];

	for &held in &game.hand[player] {
		let b = book(held);

		if !game.resolved[b] {
			represented[b] = true;
		}
	}

	for target in 0..game.active {
		if target == player
			|| game.hand[target].is_empty()
			|| game.team[target] == game.team[player]
		{
			continue;
		}

		for card in 0..CARDS {
			let b = book(card);

			if !represented[b]
				|| game.resolved[b]
				|| game.hand[player].contains(&card)
				|| belief.known_misses[target][card]
				|| belief.possible[card]
					& (1u64 << target)
					== 0
			{
				continue;
			}

			asks.push(Ask { target, card });
		}
	}

	asks
}

fn hit_probability(
	belief: &Belief,
	ask: Ask,
) -> f64 {
	let mask = belief.possible[ask.card];
	let count = mask.count_ones();

	if count == 0
		|| mask & (1u64 << ask.target) == 0
	{
		return 0.0;
	}

	1.0 / count as f64
}

fn expected_information_gain(
	belief: &Belief,
	ask: Ask,
) -> f64 {
	let count =
		belief.possible[ask.card].count_ones();

	if count <= 1 {
		return 0.0;
	}

	let p = hit_probability(belief, ask);

	let before = log2(count as f64);

	let expected_after =
		(1.0 - p) * log2((count - 1) as f64);

	let location_gain =
		(before - expected_after).max(0.0);

	let answer_gain = binary_entropy(p);

	location_gain + answer_gain
}

fn choose_ask(
	game: &Game,
	player: usize,
	belief: &Belief,
) -> Option<Ask> {
	legal_asks(game, player, belief)
		.into_iter()
		.max_by(|a, b| {
			let a_score =
				expected_information_gain(
					belief,
					*a,
				);

			let b_score =
				expected_information_gain(
					belief,
					*b,
				);

			match a_score.partial_cmp(&b_score) {
				Some(Ordering::Equal) | None => b
					.card
					.cmp(&a.card)
					.then_with(|| {
						b.target.cmp(&a.target)
					}),
				Some(order) => order,
			}
		})
}

fn claim_assignment(
	game: &Game,
	player: usize,
	belief: &Belief,
	b: usize,
) -> Option<[usize; BOOK_SIZE]> {
	let mut assignment = [player; BOOK_SIZE];

	for offset in 0..BOOK_SIZE {
		let card = b * BOOK_SIZE + offset;
		let mask = belief.possible[card];

		if mask == 0 || mask.count_ones() != 1 {
			return None;
		}

		let holder =
			mask.trailing_zeros() as usize;

		if holder >= game.active
			|| game.team[holder]
				!= game.team[player]
		{
			return None;
		}

		assignment[offset] = holder;
	}

	Some(assignment)
}

fn claim_all_in_hand(
	game: &Game,
	player: usize,
	b: usize,
) -> bool {
	for card in
		b * BOOK_SIZE..(b + 1) * BOOK_SIZE
	{
		if !game.hand[player].contains(&card) {
			return false;
		}
	}

	true
}

fn known_certain_claim(
	game: &Game,
	player: usize,
	belief: &Belief,
) -> Option<usize> {
	for b in 0..BOOKS {
		if game.resolved[b] {
			continue;
		}

		if claim_all_in_hand(game, player, b)
			|| claim_assignment(
				game,
				player,
				belief,
				b,
			)
			.is_some()
		{
			return Some(b);
		}
	}

	None
}

fn expected_claim_value(
	game: &Game,
	player: usize,
	belief: &Belief,
	b: usize,
) -> f64 {
	if game.resolved[b] {
		return f64::NEG_INFINITY;
	}

	let own_team = game.team[player];
	let mut probability = 1.0;

	for card in
		b * BOOK_SIZE..(b + 1) * BOOK_SIZE
	{
		let mask = belief.possible[card];
		let count = mask.count_ones();

		if count == 0 {
			return f64::NEG_INFINITY;
		}

		let mut team_count = 0usize;

		for holder in 0..game.active {
			if mask & (1u64 << holder) != 0
				&& game.team[holder]
					== own_team
			{
				team_count += 1;
			}
		}

		if team_count == 0 {
			return 0.0;
		}

		probability *=
			team_count as f64 / count as f64;
	}

	probability
}

fn choose_fallback_claim(
	game: &Game,
	player: usize,
	belief: &Belief,
) -> usize {
	(0..BOOKS)
		.filter(|&b| !game.resolved[b])
		.max_by(|&a, &b| {
			expected_claim_value(
				game,
				player,
				belief,
				a,
			)
			.partial_cmp(
				&expected_claim_value(
					game,
					player,
					belief,
					b,
				),
			)
			.unwrap_or(Ordering::Equal)
			.then_with(|| b.cmp(&a))
		})
		.expect("no unresolved half-suit")
}

fn resolve_claim(
	game: &mut Game,
	player: usize,
	b: usize,
	beliefs: &mut [Belief],
) {
	let start = b * BOOK_SIZE;
	let end = start + BOOK_SIZE;

	let mut assignment = [player; BOOK_SIZE];

	if let Some(exact) = claim_assignment(
		game,
		player,
		&beliefs[player],
		b,
	) {
		assignment = exact;
	} else {
		for offset in 0..BOOK_SIZE {
			let card = start + offset;
			let mask =
				beliefs[player].possible[card];

			let mut best_holder = player;
			let mut best_probability =
				f64::NEG_INFINITY;

			for holder in 0..game.active {
				if game.team[holder]
					!= game.team[player]
					|| mask & (1u64 << holder)
						== 0
				{
					continue;
				}

				let count = mask.count_ones();

				if count == 0 {
					continue;
				}

				let probability =
					1.0 / count as f64;

				if probability
					> best_probability
				{
					best_probability =
						probability;
					best_holder = holder;
				}
			}

			assignment[offset] = best_holder;
		}
	}

	let mut opponent_card = false;
	let mut all_correct = true;

	for offset in 0..BOOK_SIZE {
		let card = start + offset;
		let actual_holder = game.owner[card];

		if game.team[actual_holder]
			!= game.team[player]
		{
			opponent_card = true;
		}

		if assignment[offset]
			!= actual_holder
		{
			all_correct = false;
		}
	}

	if opponent_card {
		let opponent_team =
			1usize - game.team[player] as usize;

		game.score[opponent_team] += 1;
	} else if all_correct {
		game.score[
			game.team[player] as usize
		] += 1;
	}

	for card in start..end {
		let owner = game.owner[card];

		game.hand[owner]
			.retain(|&c| c != card);
	}

	game.resolved[b] = true;

	for belief in beliefs.iter_mut() {
		for card in start..end {
			belief.possible[card] = 0;
		}
	}

	if game.hand[player].is_empty() {
		if let Some(teammate) =
			(0..game.active).find(|&p| {
				game.team[p]
					== game.team[player]
					&& !game.hand[p].is_empty()
			})
		{
			game.turn = teammate;
		} else if let Some(next) =
			next_nonempty(game, player)
		{
			game.turn = next;
		}
	}
}

fn all_resolved(game: &Game) -> bool {
	game.resolved.iter().all(|&x| x)
}

fn team_has_cards(
	game: &Game,
	team: u8,
) -> bool {
	(0..game.active).any(|p| {
		game.team[p] == team
			&& !game.hand[p].is_empty()
	})
}

fn process_question(
	game: &mut Game,
	asker: usize,
	ask: Ask,
	beliefs: &mut [Belief],
) -> f64 {
	let before_entropy =
		memory_entropy(
			game,
			&beliefs[asker],
		);

	let before_card =
		beliefs[asker]
			.possible[ask.card]
			.count_ones();

	let p = hit_probability(
		&beliefs[asker],
		ask,
	);

	let answer_information =
		binary_entropy(p);

	let actual_hit =
		game.owner[ask.card] == ask.target;

	let outcome = if actual_hit {
		Outcome::Hit
	} else {
		Outcome::Miss
	};

	if actual_hit {
		game.hand[ask.target]
			.retain(|&c| c != ask.card);

		game.hand[asker].push(ask.card);
		game.owner[ask.card] = asker;
	}

	for observer in 0..game.active {
		update_belief_after_question(
			game,
			&mut beliefs[observer],
			ask,
			outcome,
			asker,
			observer,
		);
	}

	let after_card =
		beliefs[asker]
			.possible[ask.card]
			.count_ones();

	let location_gain =
		if before_card > 0 && after_card > 0 {
			(
				log2(before_card as f64)
					- log2(after_card as f64)
			)
			.max(0.0)
		} else {
			0.0
		};

	let after_entropy =
		memory_entropy(
			game,
			&beliefs[asker],
		);

	let entropy_reduction =
		(before_entropy - after_entropy)
			.max(0.0);

	game.moves += 1;

	game.turn = if actual_hit {
		asker
	} else {
		ask.target
	};

	if game.hand[game.turn].is_empty() {
		if let Some(next) =
			next_nonempty(game, game.turn)
		{
			game.turn = next;
		}
	}

	answer_information
		+ location_gain
		+ entropy_reduction
}

fn play_game(
	n: usize,
	rng: &mut impl rand::Rng,
) -> Result<
	(f64, f64, f64, f64, f64),
	String,
> {
	let mut game = setup(n, rng);

	let mut beliefs =
		(0..game.active)
			.map(|player| {
				initial_belief(&game, player)
			})
			.collect::<Vec<_>>();

	let mut processing_total = 0.0;
	let mut memory_total = 0.0;
	let mut memory_samples = 0usize;

	while !all_resolved(&game) {
		if game.moves >= MAX_MOVES {
			return Err(format!(
				"MAX_MOVES reached: n={}, moves={}, turn={}, unresolved={:?}",
				n,
				game.moves,
				game.turn,
				game.resolved
			));
		}

		let player = game.turn;

		if game.hand[player].is_empty() {
			if let Some(next) =
				next_nonempty(&game, player)
			{
				game.turn = next;
				continue;
			}

			return Err(format!(
				"no nonempty player available: n={}, moves={}",
				n,
				game.moves
			));
		}

		for observer in 0..game.active {
			memory_total +=
				memory_entropy(
					&game,
					&beliefs[observer],
				);

			memory_samples += 1;
		}

		if let Some(b) =
			known_certain_claim(
				&game,
				player,
				&beliefs[player],
			)
		{
			let before =
				memory_entropy(
					&game,
					&beliefs[player],
				);

			resolve_claim(
				&mut game,
				player,
				b,
				&mut beliefs,
			);

			let after =
				memory_entropy(
					&game,
					&beliefs[player],
				);

			processing_total +=
				(before - after).max(0.0);

			game.moves += 1;
			continue;
		}

		let team0_alive =
			team_has_cards(&game, 0);

		let team1_alive =
			team_has_cards(&game, 1);

		if !team0_alive || !team1_alive {
			let b =
				choose_fallback_claim(
					&game,
					player,
					&beliefs[player],
				);

			let before =
				memory_entropy(
					&game,
					&beliefs[player],
				);

			resolve_claim(
				&mut game,
				player,
				b,
				&mut beliefs,
			);

			let after =
				memory_entropy(
					&game,
					&beliefs[player],
				);

			processing_total +=
				(before - after).max(0.0);

			game.moves += 1;
			continue;
		}

		if let Some(ask) =
			choose_ask(
				&game,
				player,
				&beliefs[player],
			)
		{
			processing_total +=
				process_question(
					&mut game,
					player,
					ask,
					&mut beliefs,
				);

			continue;
		}

		let b =
			choose_fallback_claim(
				&game,
				player,
				&beliefs[player],
			);

		let before =
			memory_entropy(
				&game,
				&beliefs[player],
			);

		resolve_claim(
			&mut game,
			player,
			b,
			&mut beliefs,
		);

		let after =
			memory_entropy(
				&game,
				&beliefs[player],
			);

		processing_total +=
			(before - after).max(0.0);

		game.moves += 1;
	}

	let active = game.active as f64;

	let processing_game =
		processing_total;

	let processing_game_per_player =
		processing_total / active;

	let processing_turn =
		if game.moves == 0 {
			0.0
		} else {
			processing_total
				/ game.moves as f64
		};

	let memory_player =
		if memory_samples == 0 {
			0.0
		} else {
			memory_total
				/ memory_samples as f64
		};

	Ok((
		processing_game,
		processing_game_per_player,
		processing_turn,
		memory_player,
		game.moves as f64,
	))
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
	if values.is_empty() {
		return (0.0, 0.0);
	}

	let mean =
		values.iter().sum::<f64>()
			/ values.len() as f64;

	let variance =
		values
			.iter()
			.map(|x| (x - mean).powi(2))
			.sum::<f64>()
			/ values.len() as f64;

	(mean, variance.sqrt())
}

fn benchmark(
	n: usize,
) -> Result<ResultRow, String> {
	let mut rng = thread_rng();

	let mut processing_game =
		Vec::with_capacity(ITERATIONS);

	let mut processing_game_per_player =
		Vec::with_capacity(ITERATIONS);

	let mut processing_turn =
		Vec::with_capacity(ITERATIONS);

	let mut memory_player =
		Vec::with_capacity(ITERATIONS);

	let mut moves =
		Vec::with_capacity(ITERATIONS);

	for iteration in 0..ITERATIONS {
		let result =
			play_game(n, &mut rng)?;

		processing_game.push(result.0);
		processing_game_per_player
			.push(result.1);
		processing_turn.push(result.2);
		memory_player.push(result.3);
		moves.push(result.4);

		print!(
			"\r  n = {:>2}   game {:>3}/{:<3}",
			n,
			iteration + 1,
			ITERATIONS
		);

		io::stdout()
			.flush()
			.map_err(|e| e.to_string())?;
	}

	println!();

	let (
		processing_game_mean,
		processing_game_sd,
	) = mean_sd(&processing_game);

	let (
		processing_game_per_player_mean,
		processing_game_per_player_sd,
	) = mean_sd(
		&processing_game_per_player,
	);

	let (
		processing_turn_mean,
		processing_turn_sd,
	) = mean_sd(&processing_turn);

	let (
		memory_player_mean,
		memory_player_sd,
	) = mean_sd(&memory_player);

	let (moves_mean, moves_sd) =
		mean_sd(&moves);

	Ok(ResultRow {
		n,
		active: active_players(n),
		processing_game:
			processing_game_mean,
		processing_game_sd,
		processing_game_per_player:
			processing_game_per_player_mean,
		processing_game_per_player_sd,
		processing_turn:
			processing_turn_mean,
		processing_turn_sd,
		memory_player:
			memory_player_mean,
		memory_player_sd,
		moves: moves_mean,
		moves_sd,
	})
}

fn print_table(rows: &[ResultRow]) {
	println!();

	println!(
		"+------+--------+--------------+--------------+--------------+--------------+--------------+"
	);

	println!(
		"| {:^4} | {:^6} | {:^12} | {:^12} | {:^12} | {:^12} | {:^12} |",
		"n",
		"active",
		"proc/game",
		"proc/player",
		"proc/turn",
		"memory/player",
		"moves"
	);

	println!(
		"+------+--------+--------------+--------------+--------------+--------------+--------------+"
	);

	for row in rows {
		println!(
			"| {:>4} | {:>6} | {:>12.3} | {:>12.3} | {:>12.3} | {:>12.3} | {:>12.2} |",
			row.n,
			row.active,
			row.processing_game,
			row.processing_game_per_player,
			row.processing_turn,
			row.memory_player,
			row.moves
		);
	}

	println!(
		"+------+--------+--------------+--------------+--------------+--------------+--------------+"
	);

	println!();
}

fn write_csv(
	rows: &[ResultRow],
) -> io::Result<()> {
	let file =
		File::create(
			"literature_benchmark.csv",
		)?;

	let mut writer =
		BufWriter::new(file);

	writeln!(
		writer,
		"n,active_players,processing_game,processing_game_sd,processing_game_per_player,processing_game_per_player_sd,processing_turn,processing_turn_sd,memory_player,memory_player_sd,moves,moves_sd"
	)?;

	for row in rows {
		writeln!(
			writer,
			"{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
			row.n,
			row.active,
			row.processing_game,
			row.processing_game_sd,
			row.processing_game_per_player,
			row.processing_game_per_player_sd,
			row.processing_turn,
			row.processing_turn_sd,
			row.memory_player,
			row.memory_player_sd,
			row.moves,
			row.moves_sd
		)?;
	}

	Ok(())
}

fn y_range(
	rows: &[ResultRow],
	value: fn(&ResultRow) -> f64,
) -> (f64, f64) {
	let mut min = f64::INFINITY;
	let mut max = f64::NEG_INFINITY;

	for row in rows {
		let v = value(row);

		min = min.min(v);
		max = max.max(v);
	}

	if !min.is_finite() || !max.is_finite() {
		return (0.0, 1.0);
	}

	if (max - min).abs() < 1e-12 {
		if max <= 0.0 {
			return (0.0, 1.0);
		}

		return (0.0, max * 1.1);
	}

	let padding =
		(max - min) * 0.08;

	(
		(min - padding).max(0.0),
		max + padding,
	)
}

fn draw_graph(
	filename: &str,
	title: &str,
	y_desc: &str,
	rows: &[ResultRow],
	value: fn(&ResultRow) -> f64,
) -> Result<(), Box<dyn std::error::Error>> {
	let root =
		SVGBackend::new(
			filename,
			(1200, 700),
		)
		.into_drawing_area();

	root.fill(&WHITE)?;

	let (y_min, y_max) =
		y_range(rows, value);

	let mut chart =
		ChartBuilder::on(&root)
			.caption(
				title,
				("sans-serif", 30),
			)
			.margin(30)
			.x_label_area_size(55)
			.y_label_area_size(90)
			.build_cartesian_2d(
				4usize..60usize,
				y_min..y_max,
			)?;

	chart
		.configure_mesh()
		.x_desc(
			"Total requested players",
		)
		.y_desc(y_desc)
		.x_labels(15)
		.draw()?;

	chart.draw_series(
		LineSeries::new(
			rows.iter()
				.map(|r| (r.n, value(r))),
			&BLUE,
		),
	)?;

	chart.draw_series(
		rows.iter().map(|r| {
			Circle::new(
				(r.n, value(r)),
				4,
				BLUE.filled(),
			)
		}),
	)?;

	root.present()?;

	Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	println!();
	println!(
		"================================================================================================================"
	);
	println!(
		"                                      LITERATURE BENCHMARK"
	);
	println!(
		"================================================================================================================"
	);
	println!();
	println!(
		"  Iterations per n : {}",
		ITERATIONS
	);
	println!(
		"  n                 : 4, 6, ..., 60"
	);
	println!(
		"  Active players    : min(n, 48)"
	);
	println!(
		"  proc/game         : total abstract processing cost"
	);
	println!(
		"  proc/player       : total abstract processing cost / active players"
	);
	println!(
		"  proc/turn         : total abstract processing cost / actual turns"
	);
	println!(
		"  memory/player     : average Shannon memory / active player"
	);
	println!(
		"  moves             : actual turns"
	);
	println!();

	let mut rows = Vec::new();

	for n in (4..=60).step_by(2) {
		let row =
			benchmark(n).map_err(
				|e| {
					format!(
						"Benchmark failed for n={}: {}",
						n,
						e
					)
				},
			)?;

		rows.push(row);
	}

	print_table(&rows);

	write_csv(&rows)?;

	draw_graph(
		"literature_processing_game.svg",
		"Literature: Total Abstract Processing Cost",
		"Information bits / game",
		&rows,
		|r| r.processing_game,
	)?;

	draw_graph(
		"literature_processing_player.svg",
		"Literature: Abstract Processing Cost per Player",
		"Information bits / player / game",
		&rows,
		|r| {
			r.processing_game_per_player
		},
	)?;

	draw_graph(
		"literature_processing_turn.svg",
		"Literature: Abstract Processing Cost per Turn",
		"Information bits / turn",
		&rows,
		|r| r.processing_turn,
	)?;

	draw_graph(
		"literature_memory.svg",
		"Literature: Abstract Memory Requirement",
		"Shannon bits / player",
		&rows,
		|r| r.memory_player,
	)?;

	draw_graph(
		"literature_moves.svg",
		"Literature: Average Game Length",
		"Actual turns / game",
		&rows,
		|r| r.moves,
	)?;

	println!(
		"Generated:"
	);
	println!(
		"  literature_benchmark.csv"
	);
	println!(
		"  literature_processing_game.svg"
	);
	println!(
		"  literature_processing_player.svg"
	);
	println!(
		"  literature_processing_turn.svg"
	);
	println!(
		"  literature_memory.svg"
	);
	println!(
		"  literature_moves.svg"
	);
	println!();

	Ok(())
}
