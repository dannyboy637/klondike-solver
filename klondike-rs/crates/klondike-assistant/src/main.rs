mod app;
mod game_state;
mod input;
mod render;
mod solver;

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::prelude::CrosstermBackend;
use ratatui::Terminal;

use app::App;

fn main() -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let result = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    if let Err(err) = result {
        eprintln!("Error: {}", err);
    }

    Ok(())
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> anyhow::Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|f| render::render(f, &app))?;

        if app.should_quit {
            break;
        }

        // Poll for events with a timeout so we can refresh
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    KeyCode::Esc => {
                        app.should_quit = true;
                    }
                    KeyCode::Enter => {
                        app.on_enter();
                    }
                    KeyCode::Backspace => {
                        app.on_backspace();
                    }
                    KeyCode::Char(c) => {
                        // Check for special keys first (only when input buffer is empty)
                        if app.input_buffer.is_empty() {
                            match c {
                                'q' | 'u' | 'r' | 'n' => {
                                    app.on_key(c);
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        app.on_char(c);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
