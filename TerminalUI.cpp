#include "TerminalUI.h"
#include <ncurses.h>
#include <panel.h>
#include <thread>
#include <atomic>
#include <cmath>
#include <sstream>
#include <iomanip>

// Stałe konfiguracyjne
#define REFRESH_RATE 60
#define COLOR_BTC 8
#define COLOR_HEAT 9

class TerminalUI::Impl {
public:
    Impl(StatsProvider& stats) : stats(stats), running(true) {
        InitNCurses();
        StartInputThread();
        StartRenderThread();
    }

    ~Impl() {
        running = false;
        if(renderThread.joinable()) renderThread.join();
        if(inputThread.joinable()) inputThread.join();
        EndNCurses();
    }

private:
    void InitNCurses() {
        initscr();
        start_color();
        curs_set(0);
        noecho();
        keypad(stdscr, TRUE);
        mousemask(ALL_MOUSE_EVENTS, NULL);
        timeout(0);

        // Inicjalizacja kolorów
        init_pair(COLOR_BTC, COLOR_YELLOW, COLOR_BLACK);
        init_pair(COLOR_HEAT, COLOR_RED, COLOR_BLACK);
        init_pair(2, COLOR_GREEN, COLOR_BLACK);
        init_pair(3, COLOR_CYAN, COLOR_BLACK);
        
        // Rejestracja kolorów niestandardowych
        if(can_change_color()) {
            init_color(COLOR_BTC, 1000, 580, 0);    // Kolor Bitcoin
            init_color(COLOR_HEAT, 1000, 0, 0);      // Czerwony heatmap
        }

        // Utworzenie okien
        int height, width;
        getmaxyx(stdscr, height, width);
        
        mainWin = newwin(height-5, width, 0, 0);
        logWin = newwin(5, width/2, height-5, 0);
        statWin = newwin(5, width/2, height-5, width/2);

        // Włącz przewijanie dla okna logów
        scrollok(logWin, TRUE);
    }

    void EndNCurses() {
        delwin(mainWin);
        delwin(logWin);
        delwin(statWin);
        endwin();
    }

    void StartRenderThread() {
        renderThread = std::thread([this]() {
            while(running) {
                auto start = std::chrono::high_resolution_clock::now();
                Render();
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000/REFRESH_RATE) - elapsed);
            }
        });
    }

    void StartInputThread() {
        inputThread = std::thread([this]() {
            while(running) {
                HandleInput();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    void Render() {
        std::lock_guard<std::mutex> lock(uiMutex);
        RenderMainWindow();
        RenderStatsWindow();
        RenderLogWindow();
        RefreshAll();
    }

    void RenderMainWindow() {
        werase(mainWin);
        
        // Rysowanie wykresu w czasie rzeczywistym
        DrawHistoryGraph();
        
        // Rysowanie heatmapy
        DrawHeatmap();
        
        // Rysowanie logo Bitcoin
        DrawBitcoinLogo();
        
        wrefresh(mainWin);
    }

    void DrawHistoryGraph() {
        int width, height;
        getmaxyx(mainWin, height, width);
        
        auto history = stats.GetSpeedHistory();
        if(history.empty()) return;

        int graphHeight = height - 4;
        int graphWidth = width - 20;
        
        // Obliczanie skalowania
        double maxVal = *std::max_element(history.begin(), history.end());
        double yScale = graphHeight / (maxVal * 1.1);
        
        // Rysowanie osi
        mvwaddch(mainWin, graphHeight+1, 0, ACS_LLCORNER);
        mvwhline(mainWin, graphHeight+1, 1, ACS_HLINE, graphWidth);
        mvwvline(mainWin, 0, graphWidth+1, ACS_VLINE, graphHeight+1);
        mvwaddch(mainWin, graphHeight+1, graphWidth+1, ACS_LRCORNER);

        // Rysowanie danych
        for(size_t i = 0; i < history.size() && i < (size_t)graphWidth; ++i) {
            int barHeight = std::ceil(history[i] * yScale);
            for(int j = 0; j < barHeight; ++j) {
                mvwaddch(mainWin, graphHeight - j, i + 1, ACS_CKBOARD | COLOR_PAIR(COLOR_HEAT));
            }
        }
    }

    void DrawHeatmap() {
        int width, height;
        getmaxyx(mainWin, height, width);
        
        const int heatmapSize = 16;
        auto heatmap = stats.GetGPUHeatmap();
        
        int startX = width - 18;
        int startY = 1;
        
        for(int y = 0; y < heatmapSize; ++y) {
            for(int x = 0; x < heatmapSize; ++x) {
                int temp = heatmap[y*heatmapSize + x];
                int color = temp > 80 ? COLOR_HEAT : temp > 70 ? COLOR_RED : COLOR_GREEN;
                mvwaddch(mainWin, startY + y, startX + x, 
                        ACS_CKBOARD | COLOR_PAIR(color) | ((temp/10) << 8));
            }
        }
    }

    void DrawBitcoinLogo() {
        const char* logo[] = {
            "  ____  _  _     _            _    ",
            " | __ )(_)| |_  | |__   _ __ | | __",
            " |  _ \\| || __| | '_ \\ | '__|| |/ /",
            " | |_) | || |_  | | | || |   |   < ",
            " |____/|_| \\__| |_| |_||_|   |_|\\_\\"
        };
        
        wattron(mainWin, COLOR_PAIR(COLOR_BTC));
        for(int i = 0; i < 5; ++i) {
            mvwprintw(mainWin, i+1, 2, "%s", logo[i]);
        }
        wattroff(mainWin, COLOR_PAIR(COLOR_BTC));
    }

    void RenderStatsWindow() {
        werase(statWin);
        
        auto statsData = stats.GetCurrentStats();
        int line = 1;
        
        // Ramka
        box(statWin, 0, 0);
        mvwprintw(statWin, 0, 2, " STATYSTYKI ");
        
        // Dane
        mvwprintw(statWin, line++, 1, "Prędkość:   %s/s", 
                FormatNumber(statsData.speed).c_str());
        mvwprintw(statWin, line++, 1, "Sprawdzono: %s", 
                FormatNumber(statsData.total).c_str());
        mvwprintw(statWin, line++, 1, "GPU Temp:   %d°C", 
                statsData.gpuTemp);
        mvwprintw(statWin, line++, 1, "Zużycie CPU: %d%%", 
                statsData.cpuUsage);
        
        wrefresh(statWin);
    }

    void RenderLogWindow() {
        werase(logWin);
        
        // Ramka
        box(logWin, 0, 0);
        mvwprintw(logWin, 0, 2, " LOGI ");
        
        // Ostatnie 4 komunikaty
        auto logs = stats.GetRecentLogs(4);
        for(size_t i = 0; i < logs.size(); ++i) {
            mvwprintw(logWin, i+1, 1, "%s", logs[i].c_str());
        }
        
        wrefresh(logWin);
    }

    void RefreshAll() {
        wnoutrefresh(stdscr);
        wnoutrefresh(mainWin);
        wnoutrefresh(logWin);
        wnoutrefresh(statWin);
        doupdate();
    }

    void HandleInput() {
        int ch = getch();
        MEVENT event;
        
        while((ch = getch()) != ERR) {
            switch(ch) {
                case KEY_MOUSE:
                    if(getmouse(&event) == OK) {
                        HandleMouse(event);
                    }
                    break;
                case 'q':
                    running = false;
                    break;
                case KEY_RESIZE:
                    HandleResize();
                    break;
            }
        }
    }

    void HandleMouse(MEVENT& event) {
        // Obsługa kliknięć w interfejsie
        if(event.bstate & BUTTON1_CLICKED) {
            if(wenclose(logWin, event.y, event.x)) {
                stats.ToggleLogLevel();
            }
        }
    }

    void HandleResize() {
        std::lock_guard<std::mutex> lock(uiMutex);
        endwin();
        refresh();
        InitNCurses();
    }

    std::string FormatNumber(uint64_t number) {
        std::stringstream ss;
        if(number >= 1'000'000'000) {
            ss << std::fixed << std::setprecision(1) << (number/1'000'000'000.0) << "G";
        } else if(number >= 1'000'000) {
            ss << std::fixed << std::setprecision(1) << (number/1'000'000.0) << "M";
        } else if(number >= 1'000) {
            ss << std::fixed << std::setprecision(1) << (number/1'000.0) << "K";
        } else {
            ss << number;
        }
        return ss.str();
    }

    // Członkowie klasy
    StatsProvider& stats;
    std::atomic<bool> running;
    std::thread renderThread;
    std::thread inputThread;
    std::mutex uiMutex;
    
    WINDOW *mainWin, *logWin, *statWin;
};

// Interfejs publiczny
TerminalUI::TerminalUI(StatsProvider& stats) : impl(new Impl(stats)) {}
TerminalUI::~TerminalUI() = default;
