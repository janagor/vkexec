#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fmt/base.h>
#include <fmt/format.h>
#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <functional>
#include <memory>
#include <optional>

#include <random>

#include <CLI/CLI.hpp>
#include <ftxui/component/component.hpp>// for Slider
#include <ftxui/component/screen_interactive.hpp>// for ScreenInteractive
#include <spdlog/spdlog.h>

#include <lefticus/tools/non_promoting_ints.hpp>

// This file will be generated automatically when cur_you run the CMake
// configuration step. It creates a namespace called `myproject`. You can modify
// the source template at `configured_files/config.hpp.in`.
#include <internal_use_only/config.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

template<std::size_t Width, std::size_t Height> struct GameBoard
{
  static constexpr std::size_t kWidth = Width;
  static constexpr std::size_t kHeight = Height;

  std::array<std::array<std::string, kHeight>, kWidth> strings;
  std::array<std::array<bool, kHeight>, kWidth> values{};

  std::size_t move_count{ 0 };

  auto get_string(std::size_t cur_x, std::size_t cur_y) -> std::string & { return strings.at(cur_x).at(cur_y); }


  void set(std::size_t cur_x, std::size_t cur_y, bool new_value)
  {
    get(cur_x, cur_y) = new_value;

    if (new_value) {
      get_string(cur_x, cur_y) = " ON";
    } else {
      get_string(cur_x, cur_y) = "OFF";
    }
  }

  void visit(auto visitor)
  {
    for (std::size_t cur_x = 0; cur_x < kWidth; ++cur_x) {
      for (std::size_t cur_y = 0; cur_y < kHeight; ++cur_y) { visitor(cur_x, cur_y, *this); }
    }
  }

  [[nodiscard]] auto get(std::size_t cur_x, std::size_t cur_y) const -> bool { return values.at(cur_x).at(cur_y); }

  [[nodiscard]] auto get(std::size_t cur_x, std::size_t cur_y) -> bool & { return values.at(cur_x).at(cur_y); }

  GameBoard()
  {
    visit([](const auto cur_x, const auto cur_y, auto &gameboard) -> auto { gameboard.set(cur_x, cur_y, true); });
  }

  void update_strings()
  {
    for (std::size_t cur_x = 0; cur_x < kWidth; ++cur_x) {
      for (std::size_t cur_y = 0; cur_y < kHeight; ++cur_y) { set(cur_x, cur_y, get(cur_x, cur_y)); }
    }
  }

  void toggle(std::size_t cur_x, std::size_t cur_y) { set(cur_x, cur_y, !get(cur_x, cur_y)); }

  void press(std::size_t cur_x, std::size_t cur_y)
  {
    ++move_count;
    toggle(cur_x, cur_y);
    if (cur_x > 0) { toggle(cur_x - 1, cur_y); }
    if (cur_y > 0) { toggle(cur_x, cur_y - 1); }
    if (cur_x < kWidth - 1) { toggle(cur_x + 1, cur_y); }
    if (cur_y < kHeight - 1) { toggle(cur_x, cur_y + 1); }
  }

  [[nodiscard]] auto solved() const -> bool
  {
    for (std::size_t cur_x = 0; cur_x < kWidth; ++cur_x) {
      for (std::size_t cur_y = 0; cur_y < kHeight; ++cur_y) {
        if (!get(cur_x, cur_y)) { return false; }
      }
    }

    return true;
  }
};

namespace {

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
auto MakeConsequenceLayout(const std::vector<ftxui::Component> &buttons,
  const ftxui::Component &quit_button, std::size_t board_width, std::size_t board_height) -> ftxui::Element
// NOLINTEND(bugprone-easily-swappable-parameters)
{
  std::vector<ftxui::Element> rows;

  std::size_t idx = 0;

  for (std::size_t cur_x = 0; cur_x < board_width; ++cur_x) {
    std::vector<ftxui::Element> row;
    for (std::size_t cur_y = 0; cur_y < board_height; ++cur_y) {
      row.push_back(buttons.at(idx)->Render());
      ++idx;
    }
    rows.push_back(ftxui::hbox(std::move(row)));
  }

  rows.push_back(ftxui::hbox({ quit_button->Render() }));

  return ftxui::vbox(std::move(rows));
}

void ConsequenceGame()
{
  auto screen = ftxui::ScreenInteractive::TerminalOutput();

  GameBoard<3, 3> game_board;

  std::string quit_text;

  const auto update_quit_text = [&quit_text](const auto &game_board_param) -> auto {
    quit_text = fmt::format("Quit ({} moves)", game_board_param.move_count);
    if (game_board_param.solved()) { quit_text += " Solved!"; }
  };

  std::vector<ftxui::Component> buttons;
  for (std::size_t cur_x = 0; cur_x < game_board.kWidth; ++cur_x) {
    for (std::size_t cur_y = 0; cur_y < game_board.kHeight; ++cur_y) {
      buttons.push_back(ftxui::Button(&game_board.get_string(cur_x, cur_y), [=, &game_board] -> void {
        if (!game_board.solved()) { game_board.press(cur_x, cur_y); }
        update_quit_text(game_board);
      }));
    }
  }

  auto quit_button = ftxui::Button(&quit_text, screen.ExitLoopClosure());

  static constexpr int kRandomizationIterations = 100;
  static constexpr int kRandomSeed = 42;

  std::mt19937 gen32{ kRandomSeed };// NOLINT fixed seed

  // NOLINTNEXTLINE This cannot be const
  std::uniform_int_distribution<std::size_t> dist_x(static_cast<std::size_t>(0), game_board.kWidth - 1);
  // NOLINTNEXTLINE This cannot be const
  std::uniform_int_distribution<std::size_t> dist_y(static_cast<std::size_t>(0), game_board.kHeight - 1);

  for (int i = 0; i < kRandomizationIterations; ++i) { game_board.press(dist_x(gen32), dist_y(gen32)); }
  game_board.move_count = 0;
  update_quit_text(game_board);

  auto all_buttons = buttons;
  all_buttons.push_back(quit_button);
  auto container = ftxui::Container::Horizontal(all_buttons);

  screen.Loop(ftxui::Renderer(container, [&] -> ftxui::Element {
    return MakeConsequenceLayout(buttons, quit_button, game_board.kWidth, game_board.kHeight);
  }));
}
}// namespace

struct Color
{
  lefticus::tools::uint_np8_t r{ static_cast<std::uint8_t>(0) };
  lefticus::tools::uint_np8_t g{ static_cast<std::uint8_t>(0) };
  lefticus::tools::uint_np8_t b{ static_cast<std::uint8_t>(0) };
};

// A simple way of representing a bitmap on screen using only characters
struct Bitmap : ftxui::Node
{
  Bitmap(std::size_t width, std::size_t height)// NOLINT same typed parameters adjacent to each other
    : width_(width), height_(height)
  {}

  auto at(std::size_t cur_x, std::size_t cur_y) -> Color & { return pixels_.at((width_ * cur_y) + cur_x); }

  void ComputeRequirement() override
  {
    requirement_.min_x = static_cast<int>(width_);
    requirement_.min_y = static_cast<int>(height_ / 2);
  }

  void Render(ftxui::Screen &screen) override
  {
    for (std::size_t cur_x = 0; cur_x < width_; ++cur_x) {
      for (std::size_t cur_y = 0; cur_y < height_ / 2; ++cur_y) {
        auto &pixel = screen.PixelAt(box_.x_min + static_cast<int>(cur_x), box_.y_min + static_cast<int>(cur_y));
        pixel.character = "▄";
        const auto &top_color = at(cur_x, cur_y * 2);
        const auto &bottom_color = at(cur_x, (cur_y * 2) + 1);
        pixel.background_color = ftxui::Color{ top_color.r.get(), top_color.g.get(), top_color.b.get() };
        pixel.foreground_color = ftxui::Color{ bottom_color.r.get(), bottom_color.g.get(), bottom_color.b.get() };
      }
    }
  }

  [[nodiscard]] auto width() const noexcept { return width_; }

  [[nodiscard]] auto height() const noexcept { return height_; }

  [[nodiscard]] auto data() noexcept -> auto & { return pixels_; }

private:
  std::size_t width_;
  std::size_t height_;

  std::vector<Color> pixels_ = std::vector<Color>(width_ * height_, Color{});
};

namespace {

auto MakeCanvasLayout(const std::shared_ptr<Bitmap> &bitmap,
  const std::shared_ptr<Bitmap> &small_bitmap,
  int &counter,
  const double &fps,
  std::chrono::steady_clock::time_point &last_time,
  const std::function<void(const std::chrono::steady_clock::duration)> &game_iteration) -> ftxui::Element
{
  const auto new_time = std::chrono::steady_clock::now();

  ++counter;
  game_iteration(new_time - last_time);
  last_time = new_time;

  return ftxui::hbox({ bitmap | ftxui::border,
    ftxui::vbox({ ftxui::text("Frame: " + std::to_string(counter)),
      ftxui::text("FPS: " + std::to_string(fps)),
      small_bitmap | ftxui::border }) });
}

void GameIterationCanvas()
{
  // this should probably have a `bitmap` helper function that does what cur_you expect
  // similar to the other parts of FTXUI
  auto bm = std::make_shared<Bitmap>(50, 50);// NOLINT magic numbers
  auto small_bm = std::make_shared<Bitmap>(6, 6);// NOLINT magic numbers

  double fps = 0;

  std::size_t max_row = 0;
  std::size_t max_col = 0;

  // to do, add total game time clock also, not just current elapsed time
  auto game_iteration = [&](const std::chrono::steady_clock::duration elapsed_time) -> void {
    // in here we simulate however much game time has elapsed. Update animations,
    // run character AI, whatever, update stats, etc

    // this isn't actually timing based for now, it's just updating the display however fast it can
    fps = 1.0
          / (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count())
             / 1'000'000.0);// NOLINT magic numbers

    for (std::size_t row = 0; row < max_row; ++row) {
      for (std::size_t col = 0; col < bm->width(); ++col) { ++(bm->at(col, row).r); }
    }

    for (std::size_t row = 0; row < bm->height(); ++row) {
      for (std::size_t col = 0; col < max_col; ++col) { ++(bm->at(col, row).g); }
    }

    // for the fun of it, let's have a second window doing interesting things
    auto &small_bm_pixel =
      small_bm->data().at(static_cast<std::size_t>(elapsed_time.count()) % small_bm->data().size());

    switch (elapsed_time.count() % 3) {
    case 0:
      small_bm_pixel.r += 11;// NOLINT Magic Number
      break;
    case 1:
      small_bm_pixel.g += 11;// NOLINT Magic Number
      break;
    case 2:
      small_bm_pixel.b += 11;// NOLINT Magic Number
      break;
    default:// literally impossible
      std::unreachable();
    }


    ++max_row;
    if (max_row >= bm->height()) { max_row = 0; }
    ++max_col;
    if (max_col >= bm->width()) { max_col = 0; }
  };

  auto screen = ftxui::ScreenInteractive::TerminalOutput();

  int counter = 0;

  auto last_time = std::chrono::steady_clock::now();

  std::atomic<bool> refresh_ui_continue = true;

  // This thread exists to make sure that the event queue has an event to
  // process at approximately a rate of 30 FPS
  std::thread refresh_ui([&] -> void {
    while (refresh_ui_continue) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1.0s / 30.0);// NOLINT magic numbers
      screen.PostEvent(ftxui::Event::Custom);
    }
  });

  screen.Loop(ftxui::Renderer(
    [&] -> ftxui::Element { return MakeCanvasLayout(bm, small_bm, counter, fps, last_time, game_iteration); }));

  refresh_ui_continue = false;
  refresh_ui.join();
}
}// namespace

// NOLINTNEXTLINE(bugprone-exception-escape)
auto main(int argc, const char **argv) -> int
{
  try {
    CLI::App app{ fmt::format("{} version {}", myproject::cmake::kProjectName, myproject::cmake::kProjectVersion) };

    std::optional<std::string> message;
    app.add_option("-m,--message", message, "A message to print back out");
    bool show_version = false;
    app.add_flag("--version", show_version, "Show version information");

    bool is_turn_based = false;
    auto *turn_based = app.add_flag("--turn_based", is_turn_based);

    bool is_loop_based = false;
    auto *loop_based = app.add_flag("--loop_based", is_loop_based);

    turn_based->excludes(loop_based);
    loop_based->excludes(turn_based);


    CLI11_PARSE(app, argc, argv);

    if (show_version) {
      fmt::print("{}\n", myproject::cmake::kProjectVersion);
      return EXIT_SUCCESS;
    }

    if (is_turn_based) {
      ConsequenceGame();
    } else {
      GameIterationCanvas();
    }

  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
  }
}
