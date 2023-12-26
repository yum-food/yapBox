#include <curses.h>
#include <ncurses.h>
#include <stdio.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

typedef std::pair<std::string, std::string> datapoint_t;

const int PAGE_LINES = 40;
const int TRANSCRIPT_CHARS = 120;

void getData(
    const std::filesystem::path& data_path,
    std::vector<datapoint_t> &datapoints,
    std::map<std::string, std::string> &transcripts) {
  datapoints.clear();
  transcripts.clear();
  printw("Scanning for files at %s\n", data_path.string().c_str());
  for (const auto& entry : std::filesystem::directory_iterator(data_path)) {
    //printw("  Checking file %s\n", entry.path().string().c_str());
    if (entry.is_regular_file()) {
      std::filesystem::path filepath = entry.path();
      std::string filename = filepath.stem().string();

      if (filepath.extension() == ".wav") {
        std::filesystem::path txt_file = filepath.replace_extension(".txt");
        if (std::filesystem::exists(txt_file)) {
          datapoints.emplace_back(filepath.string(), txt_file.string());
          std::ifstream fileStream(txt_file);
          std::stringstream buffer;
          buffer << fileStream.rdbuf();
          std::string contents = buffer.str();
          contents.erase(std::remove(contents.begin(), contents.end(), '\n'), contents.cend());
          contents.erase(std::remove(contents.begin(), contents.end(), '\r'), contents.cend());
          contents = contents.substr(0, TRANSCRIPT_CHARS);
          transcripts[txt_file.string()] = contents;
        }
      }
    }
  }
}


int main(int argc, char* argv[]) {
	const std::filesystem::path cwd = std::filesystem::current_path();
	std::filesystem::path data_path = std::filesystem::current_path();
  if (argc == 2) {
    data_path = std::filesystem::path(argv[1]);
  }

  // Initialize ncurses
  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);

  // Clear the screen and wait for 'q' or 'x'
  bool run = true;
  bool redraw = true;
  int idx = 0;
  int page_offset = 0;

  std::vector<datapoint_t> datapoints;
  std::map<std::string, std::string> transcripts;

  std::string digits;
  while (run) {
    clear();
    {
      int cur_idx = 0;
      getData(data_path, datapoints, transcripts);
      for (const auto& [txt_path, transcript] : transcripts) {
        if (cur_idx < page_offset * PAGE_LINES) {
          ++cur_idx;
          continue;
        }

        char selector = ((cur_idx % PAGE_LINES) == idx) ? '>' : ' ';
        printw("%02d %c %s: %s\n", (cur_idx % PAGE_LINES), selector, txt_path.c_str(), transcript.c_str());
        ++cur_idx;

        if (cur_idx >= (page_offset + 1) * PAGE_LINES) {
          break;
        }
      }
    }
    refresh();

    int ch = getch();
    if (ch == 'q') {
      run = false;
      continue;
    } else if (ch == 'j') {
      int step_sz = 1;
      if (digits.size() > 0) {
        step_sz = std::atoi(digits.c_str());
        digits.clear();
      }

      idx += step_sz;
      idx = std::min(PAGE_LINES - 1, idx);
    } else if (ch == 'k') {
      int step_sz = 1;
      if (digits.size() > 0) {
        step_sz = std::atoi(digits.c_str());
        digits.clear();
      }

      idx -= step_sz;
      idx = std::max(0, idx);
    } else if (ch == KEY_NPAGE) {
      ++page_offset;
    } else if (ch == KEY_PPAGE) {
      --page_offset;
      page_offset = std::max(0, page_offset);
    } else if (ch == 'x') {
      int cur_idx = 0;
      for (const auto& [txt_path, transcript] : transcripts) {
        if (cur_idx != page_offset * PAGE_LINES + idx) {
          ++cur_idx;
          continue;
        }
        std::filesystem::path wav_file = std::filesystem::path(txt_path).replace_extension(".wav");
        std::filesystem::remove(txt_path);
        std::filesystem::remove(wav_file);
        break;
      }
    } else if (ch >= '0' && ch <= '9') {
      digits += ch;
    } else if (ch == 'g') {
      int target = idx;
      if (digits.size() > 0) {
        target = std::atoi(digits.c_str());
        digits.clear();
      }

      idx = target;
      idx = std::min(PAGE_LINES - 1, idx);
      idx = std::max(0, idx);
    } else if (ch == 27) {  // ASCII value of esc key
      digits.clear();
    }
  }

  // End ncurses mode
  endwin();

  return 0;
}

