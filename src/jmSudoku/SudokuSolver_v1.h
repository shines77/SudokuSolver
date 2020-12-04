
#ifndef JM_SUDOKU_SOLVER_V1_H
#define JM_SUDOKU_SOLVER_V1_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <vector>
#include <bitset>

#include "Sudoku.h"
#include "StopWatch.h"

/************************************************

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

************************************************/

#define V1_SEARCH_MODE      SEARCH_MODE_ONE_ANSWER

namespace jmSudoku {
namespace v1 {

static const size_t kSearchMode = V1_SEARCH_MODE;

template <size_t Capacity>
struct FixedDlxNodeList {
public:
    typedef FixedDlxNodeList<Capacity> this_type;

    static const size_t kCapacity = (Capacity + 1) / 2 * 2;

    int prev[kCapacity];
    int next[kCapacity];
    int up[kCapacity];
    int down[kCapacity];
    int row[kCapacity];
    int col[kCapacity];

    FixedDlxNodeList(size_t capacity) {
    }

    ~FixedDlxNodeList() {
    }

    size_t size() const { return Capacity; }
    size_t capacity() const { return this_type::kCapacity; }
};

class DlxNodeList {
public:
    int * prev;
    int * next;
    int * up;
    int * down;
    int * row;
    int * col;

private:
    size_t size_;
    size_t capacity_;

public:
    DlxNodeList(size_t capacity)
        : prev(nullptr), next(nullptr), up(nullptr), down(nullptr),
          row(nullptr), col(nullptr), size_(0), capacity_(capacity) {
        this->init(capacity);
    }

    ~DlxNodeList() {
        this->destroy();
    }

    size_t size() const { return this->size_; }
    size_t capacity() const { return this->capacity_; }

private:
    void init(size_t capacity) {
        assert(capacity > 0);
        this->prev = new int[capacity];
        this->next = new int[capacity];
        this->up   = new int[capacity];
        this->down = new int[capacity];
        this->row  = new int[capacity];
        this->col  = new int[capacity];
    }

    void destroy() {
        if (this->prev) {
            delete[] this->prev;
            this->prev = nullptr;
        }
        if (this->next) {
            delete[] this->next;
            this->next = nullptr;
        }
        if (this->up) {
            delete[] this->up;
            this->up = nullptr;
        }
        if (this->down) {
            delete[] this->down;
            this->down = nullptr;
        }
        if (this->row) {
            delete[] this->row;
            this->row = nullptr;
        }
        if (this->col) {
            delete[] this->col;
            this->col = nullptr;
        }
    }
};

class DancingLinks {
public:
    static const size_t Rows = Sudoku::Rows;
    static const size_t Cols = Sudoku::Cols;
    static const size_t Palaces = Sudoku::Palaces;
    static const size_t Numbers = Sudoku::Numbers;

    static const size_t TotalSize = Sudoku::TotalSize;
    static const size_t TotalSize2 = Sudoku::TotalSize2;

    static size_t init_counter;
    static size_t num_guesses;
    static size_t num_no_guess;
    static size_t num_impossibles;

private:    
#if 0
    DlxNodeList         list_;
#else
    FixedDlxNodeList<Sudoku::TotalSize * 4 + 1>
                        list_;
#endif
    std::vector<int>    col_size_;
    std::vector<int>    answer_;
    int                 last_idx_;

    int rows_[TotalSize + 1];
    int cols_[TotalSize + 1];
    int numbers_[TotalSize + 1];

    SmallBitMatrix2<9, 9>  bit_rows;        // [row][num]
    SmallBitMatrix2<9, 9>  bit_cols;        // [col][num]
    SmallBitMatrix2<9, 9>  bit_palaces;     // [palace][num]

    std::vector<std::vector<int>> answers_;

    struct StackInfo {
        int index;
        int row;

        StackInfo() : index(0), row(0) {}
        StackInfo(int index, int row) : index(index), row(row) {}

        void set(int index, int row) {
            this->index = index;
            this->row = row;
        }
    };

    enum StackState {
        SearchNext,
        BackTracking,
        BackTrackingRetry,
        Last
    };

public:
    DancingLinks(size_t nodes)
        : list_(nodes), col_size_(this->cols() + 1), last_idx_(0) {
    }

    ~DancingLinks() {}

    bool is_empty() const { return (list_.next[0] == 0); }

    int cols() const { return (int)Sudoku::TotalConditions; }

    static size_t get_init_counter() { return DancingLinks::init_counter; }
    static size_t get_num_guesses() { return DancingLinks::num_guesses; }
    static size_t get_num_no_guess() { return DancingLinks::num_no_guess; }
    static size_t get_num_impossibles() { return DancingLinks::num_impossibles; }

    static size_t get_search_counter() {
        return (DancingLinks::num_guesses + DancingLinks::num_no_guess + DancingLinks::num_impossibles);
    }

    static double get_guess_percent() {
        return calc_percent(DancingLinks::num_guesses, DancingLinks::get_search_counter());
    }

    static double get_impossible_percent() {
        return calc_percent(DancingLinks::num_impossibles, DancingLinks::get_search_counter());
    }

    static double get_no_guess_percent() {
        return calc_percent(DancingLinks::num_no_guess, DancingLinks::get_search_counter());
    }

private:
    int get_min_column(int & out_min_col) const {
        int first = list_.next[0];
        int min_col = col_size_[first];
        assert(min_col >= 0);
        if (min_col <= 1) {
            out_min_col = min_col;
            return first;
        }
        int min_col_index = first;
        for (int i = list_.next[first]; i != 0 ; i = list_.next[i]) {
            int col_size = col_size_[i];
            if (col_size < min_col) {
                assert(col_size >= 0);
                if (col_size <= 1) {
                    if (col_size == 0) {
                        return 0;
                    }
                    else {
                        out_min_col = 1;
                        return i;
                    }
                }
                min_col = col_size;
                min_col_index = i;
            }
        }
        out_min_col = min_col;
        return min_col_index;
    }

    std::bitset<9> getUsable(size_t row, size_t col) {
        size_t palace = row / 3 * 3 + col / 3;
        return ~(this->bit_rows[row] | this->bit_cols[col] | this->bit_palaces[palace]);
    }

    std::bitset<9> getUsable(size_t row, size_t col, size_t palace) {
        return ~(this->bit_rows[row] | this->bit_cols[col] | this->bit_palaces[palace]);
    }

    void fillNum(size_t row, size_t col, size_t num) {
        size_t palace = row / 3 * 3 + col / 3;
        this->bit_rows[row].set(num);
        this->bit_cols[col].set(num);
        this->bit_palaces[palace].set(num);
    }

public:
    void init() {
        int cols = this->cols();
        for (int col = 0; col <= cols; col++) {
            list_.prev[col] = col - 1;
            list_.next[col] = col + 1;
            list_.up[col] = col;
            list_.down[col] = col;
            list_.row[col] = 0;
            list_.col[col] = col;
        }
        list_.prev[0] = cols;
        list_.next[cols] = 0;

        last_idx_ = cols + 1;
        for (size_t i = 0; i < col_size_.size(); i++) {
            col_size_[i] = 0;
        }

        this->bit_rows.reset();
        this->bit_cols.reset();
        this->bit_palaces.reset();

        this->answers_.clear();
        this->answer_.reserve(81);
#if (V1_SEARCH_MODE >= SEARCH_MODE_ONE_ANSWER)
        this->answers_.clear();
#endif
        init_counter = 0;
        num_guesses = 0;
        num_no_guess = 0;
        num_impossibles = 0;
    }

    void build(const std::vector<std::vector<char>> & board) {
        size_t empties = 0;
        for (size_t row = 0; row < board.size(); row++) {
            const std::vector<char> & line = board[row];
            for (size_t col = 0; col < line.size(); col++) {
                char val = line[col];
                if (val == '.') {
                    empties++;
                }
                else {
                    size_t num = val - '1';
                    this->fillNum(row, col, num);
                }
            }   
        }

        // maxRows = filled * 1 + empties * 9;
        //         = (9 * 9 - empties) * 1 + empties * 9;
        //         = (9 * 9) + empties * 8;
        size_t filled = Rows * Cols - empties;
        size_t maxRows = filled * 1 +  empties * Numbers;        

        int row_idx = 1;

        assert(Rows == board.size());
        for (size_t row = 0; row < board.size(); row++) {
            const std::vector<char> & line = board[row];
            assert(Cols == line.size());
            for (size_t col = 0; col < line.size(); col++) {
                size_t palace = row / 3 * 3 + col / 3;
                char val = line[col];
                if (val == '.') {
                    std::bitset<9> numsUsable = getUsable(row, col, palace);
                    for (size_t number = 0; number <= (Numbers - 1); number++) {
                        if (numsUsable.test(number)) {
                            int head = last_idx_;
                            int index = last_idx_;

                            this->insert(index + 0, row_idx, (int)(0      + row * 9 + col + 1));
                            this->insert(index + 1, row_idx, (int)(81 * 1 + row * 9 + number + 1));
                            this->insert(index + 2, row_idx, (int)(81 * 2 + col * 9 + number + 1));
                            this->insert(index + 3, row_idx, (int)(81 * 3 + palace * 9 + number + 1));

                            this->rows_[row_idx] = (int)row;
                            this->cols_[row_idx] = (int)col;
                            this->numbers_[row_idx] = (int)number;
                            index += 4;
                            row_idx++;

                            list_.next[index - 1] = head;
                            list_.prev[head] = index - 1;
                            last_idx_ = index;
                        }
                    }
                }
                else {
                    size_t number = val - '1';
                    int head = last_idx_;
                    int index = last_idx_;

                    this->insert(index + 0, row_idx, (int)(0      + row * 9 + col + 1));
                    this->insert(index + 1, row_idx, (int)(81 * 1 + row * 9 + number + 1));
                    this->insert(index + 2, row_idx, (int)(81 * 2 + col * 9 + number + 1));
                    this->insert(index + 3, row_idx, (int)(81 * 3 + palace * 9 + number + 1));

                    this->rows_[row_idx] = (int)row;
                    this->cols_[row_idx] = (int)col;
                    this->numbers_[row_idx] = (int)number;
                    index += 4;
                    row_idx++;

                    list_.next[index - 1] = head;
                    list_.prev[head] = index - 1;
                    last_idx_ = index;
                }                

            }
        }
        assert(row_idx <= (maxRows + 1));
    }

    void insert(int index, int row, int col) {
        list_.prev[index] = index - 1;
        list_.next[index] = index + 1;
        list_.up[index] = list_.up[col];
        list_.down[index] = col;
        list_.row[index] = row;
        list_.col[index] = col;
#if 1
        list_.down[list_.up[index]] = index;
        list_.up[col] = index;
#else
        list_.down[list_.up[index]] = index;
        list_.up[list_.down[index]] = index;
#endif
        col_size_[col]++;
    }

    void remove(int index) {
        assert(index > 0);
        int prev = list_.prev[index];
        int next = list_.next[index];
        list_.next[prev] = next;
        list_.prev[next] = prev;

        for (int row = list_.down[index]; row != index; row = list_.down[row]) {
            int up, down;
            for (int col = list_.next[row]; col != row; col = list_.next[col]) {
                up = list_.up[col];
                down = list_.down[col];
                list_.down[up] = down;
                list_.up[down] = up;
                assert(col_size_[list_.col[col]] > 0);
                col_size_[list_.col[col]]--;
            }
        }
    }

    void restore(int index) {
        assert(index > 0);
        int next = list_.next[index];
        int prev = list_.prev[index];
        list_.prev[next] = index;
        list_.next[prev] = index;

        for (int row = list_.up[index]; row != index; row = list_.up[row]) {
            for (int col = list_.prev[row]; col != row; col = list_.prev[col]) {
                int down = list_.down[col];
                int up = list_.up[col];
                list_.up[down] = col;
                list_.down[up] = col;
                col_size_[list_.col[col]]++;
            }
        }
    }

    bool search() {
        if (this->is_empty()) {
            if (kSearchMode > SearchMode::OneAnswer) {
                this->answers_.push_back(this->answer_);
                if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                    if (this->answers_.size() > 1)
                    return true;
                }
            }
            else {
                return true;
            }
        }
        
        int min_col;
        int index = get_min_column(min_col);
        if (index > 0) {
            if (min_col == 1)
                num_no_guess++;
            else
                num_guesses++;
            this->remove(index);
            for (int row = list_.down[index]; row != index; row = list_.down[row]) {
                this->answer_.push_back(list_.row[row]);
                for (int col = list_.next[row]; col != row; col = list_.next[col]) {
                    this->remove(list_.col[col]);
                }

                if (this->search()) {
                    if (kSearchMode == SearchMode::OneAnswer) {
                        return true;
                    }
                    else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                        if (this->answers_.size() > 1)
                            return true;
                    }
                }

                for (int col = list_.prev[row]; col != row; col = list_.prev[col]) {
                    this->restore(list_.col[col]);
                }
                this->answer_.pop_back();
            }
            this->restore(index);
        }
        else {
            num_impossibles++;
        }

        return false;
    }

    bool solve() {
RETRY_LOCKED_CANDIDATES:
        for (int index = list_.next[0]; index != 0 ; index = list_.next[index]) {
            size_t col_size = col_size_[index];
            if (col_size == 1) {
                assert(index > 0);

                this->remove(index);
                for (int row = list_.down[index]; row != index; row = list_.down[row]) {
                    this->answer_.push_back(list_.row[row]);
                    for (int col = list_.next[row]; col != row; col = list_.next[col]) {
                        this->remove(list_.col[col]);
                    }
                }

                init_counter++;
                goto RETRY_LOCKED_CANDIDATES;
            }
            else if (col_size == 0) {
                return false;
            }
        }

        return this->search();
    }

    void display_answer(std::vector<std::vector<char>> & board) {
        for (auto idx : this->answer_) {
            if (idx > 0) {
                board[this->rows_[idx]][this->cols_[idx]] = (char)this->numbers_[idx] + '1';
            }
        }

        Sudoku::display_board(board);
    }

    void display_answers(std::vector<std::vector<char>> & board) {
        printf("Total answers: %d\n\n", (int)this->answers_.size());
#if 0
        int i = 0;
        for (auto answer : this->answers_) {
            Sudoku::clear_board(board);
            for (auto idx : answer) {
                if (idx > 0) {
                    board[this->rows_[idx]][this->cols_[idx]] = (char)this->numbers_[idx] + '1';
                }
            }
            Sudoku::display_board(board, false, i);
            i++;
        }
#endif
    }
};

size_t DancingLinks::init_counter = 0;
size_t DancingLinks::num_guesses = 0;
size_t DancingLinks::num_no_guess = 0;
size_t DancingLinks::num_impossibles = 0;

class Solver {
public:
    typedef DancingLinks algorithm;

private:
    DancingLinks solver_;

public:
    Solver() : solver_(Sudoku::TotalSize * 4 + 1) {
    }
    ~Solver() {}

public:
    bool solve(std::vector<std::vector<char>> & board,
               double & elapsed_time,
               bool verbose = true) {
        if (verbose) {
            Sudoku::display_board(board, true);
        }

        jtest::StopWatch sw;
        sw.start();

        solver_.init();
        solver_.build(board);
        bool success = solver_.solve();

        sw.stop();
        elapsed_time = sw.getElapsedMillisec();

        if (verbose) {
            if (kSearchMode > SearchMode::OneAnswer)
                solver_.display_answers(board);
            else
                solver_.display_answer(board);
            printf("Elapsed time: %0.3f ms, init_counter: %u, recur_counter: %u\n\n"
                   "num_guesses: %u, num_impossibles: %u, no_guess: %u\n"
                   "guess%% = %0.1f %%, impossible%% = %0.1f %%, no_guess%% = %0.1f %%\n\n",
                   elapsed_time, (uint32_t)DancingLinks::get_init_counter(),
                   (uint32_t)DancingLinks::get_search_counter(),
                   (uint32_t)DancingLinks::get_num_guesses(),
                   (uint32_t)DancingLinks::get_num_impossibles(),
                   (uint32_t)DancingLinks::get_num_no_guess(),
                   DancingLinks::get_guess_percent(),
                   DancingLinks::get_impossible_percent(),
                   DancingLinks::get_no_guess_percent());
        }

        return success;
    }
};

} // namespace v1
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V1_H
