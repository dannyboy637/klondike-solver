use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, AppPhase};
use crate::game_state::GameState;
use crate::input::{format_card, foundation_label, is_red};

/// Card color styling
fn card_style(card: lonelybot::card::Card) -> Style {
    if is_red(card) {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Black)
    }
}

/// Render the full application.
pub fn render(f: &mut Frame, app: &App) {
    let size = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title bar
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Status bar
        ])
        .split(size);

    render_title(f, chunks[0], app);

    match &app.phase {
        AppPhase::Setup { .. } => render_setup(f, chunks[1], app),
        AppPhase::StockCycle { .. } => render_stock_cycle(f, chunks[1], app),
        AppPhase::Playing { .. } => render_playing(f, chunks[1], app),
        AppPhase::GameOver { won, .. } => render_game_over(f, chunks[1], *won, app),
    }

    render_status_bar(f, chunks[2], app);
}

fn render_title(f: &mut Frame, area: Rect, app: &App) {
    let phase_name = match &app.phase {
        AppPhase::Setup { .. } => "Setup",
        AppPhase::StockCycle { .. } => "Stock Cycle",
        AppPhase::Playing { .. } => "Playing",
        AppPhase::GameOver { .. } => "Game Over",
    };

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " Klondike Assistant ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" │ "),
        Span::styled(phase_name, Style::default().fg(Color::Cyan)),
        Span::raw(" │ HOP Solver"),
    ]))
    .block(Block::default().borders(Borders::ALL));

    f.render_widget(title, area);
}

fn render_setup(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Min(1),
        ])
        .split(area);

    let instructions = Paragraph::new(vec![
        Line::from("NEW GAME — Enter the 7 face-up tableau cards (left to right):"),
        Line::from("Format: rank+suit, e.g. 'ah' = A♥, 'ts' = 10♠, 'qd' = Q♦, 'kc' = K♣"),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Step 1: Tableau"),
    );

    f.render_widget(instructions, chunks[0]);

    let input_text = format!("> {}", app.input_buffer);
    let input =
        Paragraph::new(input_text).block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[1]);

    if let Some(ref err) = app.error_message {
        let error = Paragraph::new(Span::styled(err.as_str(), Style::default().fg(Color::Red)))
            .block(Block::default().borders(Borders::ALL).title("Error"));
        f.render_widget(error, chunks[2]);
    }
}

fn render_stock_cycle(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Min(1),
        ])
        .split(area);

    let instructions = Paragraph::new(vec![
        Line::from("STOCK CYCLE — Draw through the entire stock and type all 24 cards."),
        Line::from("Draw each card in the real game, then type them in order."),
        Line::from("Format: space-separated card codes, e.g. '4d 7h ks 2c ...'"),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Step 2: Stock Cards"),
    );

    f.render_widget(instructions, chunks[0]);

    let input_text = format!("> {}", app.input_buffer);
    let input =
        Paragraph::new(input_text).block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[1]);

    // Show entered tableau cards
    if let AppPhase::StockCycle { tableau_cards, .. } = &app.phase {
        let mut spans = vec![Span::raw("Tableau: ")];
        for (i, card) in tableau_cards.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw(" "));
            }
            spans.push(Span::styled(format_card(*card), card_style(*card)));
        }
        let info = Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::ALL).title("Known Cards"));
        f.render_widget(info, chunks[2]);
    }
}

fn render_playing(f: &mut Frame, area: Rect, app: &App) {
    let state = match &app.phase {
        AppPhase::Playing { state, .. } => state,
        _ => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Foundations + stock info
            Constraint::Length(12), // Tableau
            Constraint::Length(5),  // Recommendation
            Constraint::Length(3),  // Input
            Constraint::Min(1),     // Error / help
        ])
        .split(area);

    render_foundations_and_stock(f, chunks[0], state);
    render_tableau(f, chunks[1], state);
    render_recommendation(f, chunks[2], app);
    render_input(f, chunks[3], app);

    if let Some(ref err) = app.error_message {
        let error = Paragraph::new(Span::styled(err.as_str(), Style::default().fg(Color::Red)))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(error, chunks[4]);
    }
}

fn render_foundations_and_stock(f: &mut Frame, area: Rect, state: &GameState) {
    let mut spans = Vec::new();

    // Foundations
    for suit in 0..4u8 {
        if suit > 0 {
            spans.push(Span::raw("  "));
        }
        let label = foundation_label(suit, state.foundations[suit as usize]);
        let color = if suit < 2 { Color::Red } else { Color::Black };
        spans.push(Span::styled(label, Style::default().fg(color)));
    }

    spans.push(Span::raw("     "));

    // Waste info
    let waste_card = state.current_waste_card();
    spans.push(Span::raw("Waste: "));
    if let Some(card) = waste_card {
        spans.push(Span::styled(format_card(card), card_style(card)));
    } else {
        spans.push(Span::styled("--", Style::default().fg(Color::DarkGray)));
    }

    spans.push(Span::raw(format!("  ({}♦ in waste)", state.waste.len())));

    let line = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL).title("Board"));
    f.render_widget(line, area);
}

fn render_tableau(f: &mut Frame, area: Rect, state: &GameState) {
    let mut lines = Vec::new();

    // Header
    let mut header_spans = Vec::new();
    for i in 0..7u8 {
        if i > 0 {
            header_spans.push(Span::raw("  "));
        }
        header_spans.push(Span::styled(
            format!("  P{}  ", i + 1),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }
    lines.push(Line::from(header_spans));

    // Find max pile height for rendering
    let max_height = state
        .tableau
        .iter()
        .map(|p| p.total_cards())
        .max()
        .unwrap_or(0);

    let display_rows = max_height.clamp(1, 9);

    for row in 0..display_rows {
        let mut row_spans = Vec::new();
        for (col, pile) in state.tableau.iter().enumerate() {
            if col > 0 {
                row_spans.push(Span::raw("  "));
            }

            let face_down = pile.face_down as usize;
            let total = pile.total_cards();

            if row >= total {
                row_spans.push(Span::raw("      "));
            } else if row < face_down {
                row_spans.push(Span::styled("  ??  ", Style::default().fg(Color::DarkGray)));
            } else {
                let card_idx = row - face_down;
                if card_idx < pile.face_up.len() {
                    let card = pile.face_up[card_idx];
                    let card_str = format_card(card);
                    let padded = format!("{:^6}", card_str);
                    row_spans.push(Span::styled(padded, card_style(card)));
                }
            }
        }
        lines.push(Line::from(row_spans));
    }

    let tableau =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Tableau"));
    f.render_widget(tableau, area);
}

fn render_recommendation(f: &mut Frame, area: Rect, app: &App) {
    let (rec, waiting) = match &app.phase {
        AppPhase::Playing {
            recommendation,
            solving,
            ..
        } => (recommendation.as_ref(), *solving),
        _ => (None, false),
    };

    let content = if waiting {
        Line::from(Span::styled(
            "  Thinking...",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::ITALIC),
        ))
    } else if let Some(rec) = rec {
        let confidence = format!("{}/{}", rec.votes, rec.total);
        Line::from(vec![
            Span::styled(
                "  >>> ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                &rec.description,
                Style::default()
                    .fg(Color::Reset)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("[confidence: {}]", confidence),
                Style::default().fg(Color::DarkGray),
            ),
        ])
    } else {
        Line::from(Span::styled(
            "  No recommendation yet. Press Enter to get one.",
            Style::default().fg(Color::DarkGray),
        ))
    };

    let rec_widget = Paragraph::new(vec![Line::from(""), content, Line::from("")]).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Recommendation"),
    );
    f.render_widget(rec_widget, area);
}

fn render_input(f: &mut Frame, area: Rect, app: &App) {
    let awaiting_reveal = match &app.phase {
        AppPhase::Playing {
            awaiting_reveal, ..
        } => *awaiting_reveal,
        _ => None,
    };

    let prompt = if let Some(pile) = awaiting_reveal {
        format!(
            "Reveal on pile {} — type card code: > {}",
            pile + 1,
            app.input_buffer
        )
    } else {
        format!(
            "Enter=confirm  d=draw  u=undo  r=recompute  n=new  g=give up  Esc=quit  > {}",
            app.input_buffer
        )
    };

    let input = Paragraph::new(prompt).block(Block::default().borders(Borders::ALL));
    f.render_widget(input, area);
}

fn render_game_over(f: &mut Frame, area: Rect, won: bool, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(1)])
        .split(area);

    let (msg, color) = if won {
        ("YOU WON!", Color::Green)
    } else {
        ("Game Over — No moves available", Color::Red)
    };

    let result = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            msg,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("Press Esc to quit or 'n' for new game."),
    ])
    .block(Block::default().borders(Borders::ALL).title("Result"))
    .wrap(Wrap { trim: false });

    f.render_widget(result, chunks[0]);

    // Session stats
    if let AppPhase::GameOver { move_count, .. } = &app.phase {
        let stats = Paragraph::new(vec![
            Line::from(format!("Total moves: {}", move_count)),
            Line::from(format!(
                "Games this session: W{} / L{}",
                app.session_wins, app.session_losses
            )),
        ])
        .block(Block::default().borders(Borders::ALL).title("Stats"));
        f.render_widget(stats, chunks[1]);
    }
}

fn render_status_bar(f: &mut Frame, area: Rect, app: &App) {
    let move_count = match &app.phase {
        AppPhase::Playing { state, .. } => state.move_count(),
        AppPhase::GameOver { move_count, .. } => *move_count,
        _ => 0,
    };

    let unknown_count = match &app.phase {
        AppPhase::Playing { state, .. } => state.unknown_count(),
        _ => 0,
    };

    let status = Paragraph::new(Line::from(vec![
        Span::raw(format!(" Moves: {} ", move_count)),
        Span::raw("│"),
        Span::raw(format!(" Unknown: {} ", unknown_count)),
        Span::raw("│"),
        Span::raw(format!(
            " Session: W{}/L{} ",
            app.session_wins, app.session_losses
        )),
        Span::raw("│"),
        Span::styled(" n=new  Esc=quit ", Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL));

    f.render_widget(status, area);
}
