
#ifndef JM_SUDOKU_SOLVER_V1_H
#define JM_SUDOKU_SOLVER_V1_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <cstring>      // For std::memset()
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

    unsigned short prev[kCapacity];
    unsigned short next[kCapacity];
    unsigned short up[kCapacity];
    unsigned short down[kCapacity];
    unsigned short row[kCapacity];
    unsigned short col[kCapacity];

    FixedDlxNodeList(size_t capacity) {
    }

    ~FixedDlxNodeList() {
    }

    size_t size() const { return Capacity; }
    size_t capacity() const { return this_type::kCapacity; }
};

class DlxNodeList {
public:
    unsigned short * prev;
    unsigned short * next;
    unsigned short * up;
    unsigned short * down;
    unsigned short * row;
    unsigned short * col;

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
        this->prev = new unsigned short[capacity];
        this->next = new unsigned short[capacity];
        this->up   = new unsigned short[capacity];
        this->down = new unsigned short[capacity];
        this->row  = new unsigned short[capacity];
        this->col  = new unsigned short[capacity];
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

    SmallBitMatrix2<9, 9>  bit_rows;        // [row][num]
    SmallBitMatrix2<9, 9>  bit_cols;        // [col][num]
    SmallBitMatrix2<9, 9>  bit_palaces;     // [palace][num]

    unsigned short col_index_[Sudoku::TotalConditions + 1];

    unsigned short rows_[TotalSize + 1];
    unsigned short cols_[TotalSize + 1];
    unsigned short numbers_[TotalSize + 1];

    std::vector<std::vector<int>> answers_;

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
        for (int i = list_.next[first]; i != 0; i = list_.next[i]) {
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
        // size_t palace = tables.roundTo3[row] + tables.div3[col];
        return ~(this->bit_rows[row] | this->bit_cols[col] | this->bit_palaces[palace]);
    }

    std::bitset<9> getUsable(size_t row, size_t col, size_t palace) {
        return ~(this->bit_rows[row] | this->bit_cols[col] | this->bit_palaces[palace]);
    }

    void fillNum(size_t row, size_t col, size_t num) {
        size_t palace = row / 3 * 3 + col / 3;
        // size_t palace = tables.roundTo3[row] + tables.div3[col];
        this->bit_rows[row].set(num);
        this->bit_cols[col].set(num);
        this->bit_palaces[palace].set(num);
    }

public:
    int filter_unused_cols(char board[Sudoku::BoardSize]) {
        std::memset(&this->col_index_[0], 0, sizeof(this->col_index_));

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t palace_row = row / 3 * 3;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos];
                if (val != '.') {
                    size_t num = val - '1';
                    this->col_index_[0      + pos           + 1] = 0xFFFF;
                    this->col_index_[81 * 1 + row * 9 + num + 1] = 0xFFFF;
                    this->col_index_[81 * 2 + col * 9 + num + 1] = 0xFFFF;
                    size_t palace = palace_row + col / 3;
                    // size_t palace_x_9 = tables.palace_x_9[pos];
                    this->col_index_[81 * 3 + palace * 9 + num + 1] = 0xFFFF;
                }
                pos++;
            }
        }

        size_t index = 1;
        for (size_t i = 1; i < (Sudoku::TotalConditions + 1); i++) {
            if (this->col_index_[i] == 0) {
                this->col_index_[i] = (unsigned short)index;
                index++;
            }
        }
        return (int)(index - 1);
    }

    void init(char board[Sudoku::BoardSize]) {
        int cols = this->filter_unused_cols(board);
        for (int col = 0; col <= cols; col++) {
            list_.prev[col] = col - 1;
            list_.next[col] = col + 1;
            list_.up[col] = col;
            list_.down[col] = col;
            //list_.row[col] = 0;
            //list_.col[col] = col;
        }
        list_.prev[0] = cols;
        list_.next[cols] = 0;

        last_idx_ = cols + 1;
        for (int i = 0; i <= cols; i++) {
            col_size_[i] = 0;
        }

        this->bit_rows.reset();
        this->bit_cols.reset();
        this->bit_palaces.reset();

        this->answer_.clear();
        this->answer_.reserve(81);
#if (V1_SEARCH_MODE >= SEARCH_MODE_ONE_ANSWER)
        this->answers_.clear();
#endif
        init_counter = 0;
        num_guesses = 0;
        num_no_guess = 0;
        num_impossibles = 0;
    }

    void build(char board[Sudoku::BoardSize]) {
        size_t empties = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos++];
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

        pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t palace_row = row / 3 * 3;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board[pos];
                if (val == '.') {
                    size_t palace = palace_row + col / 3;
                    // size_t palace = tables.palace[pos];
                    // size_t palace_x_9 = tables.palace_x_9[pos];
                    std::bitset<9> numsUsable = getUsable(row, col, palace);
                    for (size_t number = 0; number < Numbers; number++) {
                        if (numsUsable.test(number)) {
                            int head = last_idx_;
                            int index = last_idx_;

                            this->insert(index + 0, row_idx, (int)(0      + pos              + 1));
                            this->insert(index + 1, row_idx, (int)(81 * 1 + row * 9 + number + 1));
                            this->insert(index + 2, row_idx, (int)(81 * 2 + col * 9 + number + 1));
                            this->insert(index + 3, row_idx, (int)(81 * 3 + palace * 9 + number + 1));

                            this->rows_[row_idx] = (unsigned short)row;
                            this->cols_[row_idx] = (unsigned short)col;
                            this->numbers_[row_idx] = (unsigned short)number;
                            index += 4;
                            row_idx++;

                            list_.next[index - 1] = head;
                            list_.prev[head] = index - 1;
                            last_idx_ = index;
                        }
                    }
                }
                else {
#if 0
                    size_t number = val - '1';
                    int head = last_idx_;
                    int index = last_idx_;

                    this->insert(index + 0, row_idx, (int)(0      + pos              + 1));
                    this->insert(index + 1, row_idx, (int)(81 * 1 + row * 9 + number + 1));
                    this->insert(index + 2, row_idx, (int)(81 * 2 + col * 9 + number + 1));
                    size_t palace_x_9 = tables.palace_x_9[pos];
                    this->insert(index + 3, row_idx, (int)(81 * 3 + palace_x_9 + number + 1));

                    this->rows_[row_idx] = (unsigned char)row;
                    this->cols_[row_idx] = (unsigned char)col;
                    this->numbers_[row_idx] = (unsigned char)number;
                    index += 4;
                    row_idx++;

                    list_.next[index - 1] = head;
                    list_.prev[head] = index - 1;
                    last_idx_ = index;
#endif
                }                
                pos++;
            }
        }
        assert(row_idx <= (maxRows + 1));
    }

    void insert(int index, int row, int col) {
        int save_col = col;
        col = this->col_index_[col];
        assert(col != 0xFFFF);
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
            for (int col = list_.next[row]; col != row; col = list_.next[col]) {
                int up = list_.up[col];
                int down = list_.down[col];
                list_.down[up] = down;
                list_.up[down] = up;
                assert(col_size_[list_.col[col]] > 0);
                col_size_[list_.col[col]]--;
            }
        }
    }

    void restore(int index) {
        assert(index > 0);

        for (int row = list_.up[index]; row != index; row = list_.up[row]) {
            for (int col = list_.prev[row]; col != row; col = list_.prev[col]) {
                int down = list_.down[col];
                int up = list_.up[col];
                list_.up[down] = col;
                list_.down[up] = col;
                col_size_[list_.col[col]]++;
            }
        }

        int next = list_.next[index];
        int prev = list_.prev[index];
        list_.prev[next] = index;
        list_.next[prev] = index;
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
#if 0
RETRY_LOCKED_CANDIDATES:
        for (int index = list_.next[0]; index != 0; index = list_.next[index]) {
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
#endif
        return this->search();
    }

    void display_answer(char board[Sudoku::BoardSize]) {
        for (auto idx : this->answer_) {
            if (idx > 0) {
                board[this->rows_[idx] * Rows + this->cols_[idx]] = (char)this->numbers_[idx] + '1';
            }
        }

        Sudoku::display_board(board);
    }

    void display_answers(char board[Sudoku::BoardSize]) {
        printf("Total answers: %d\n\n", (int)this->answers_.size());
        int i = 0;
        for (auto answer : this->answers_) {
            Sudoku::clear_board(board);
            for (auto idx : answer) {
                if (idx > 0) {
                    board[this->rows_[idx] * Rows + this->cols_[idx]] = (char)this->numbers_[idx] + '1';
                }
            }
            Sudoku::display_board(board, false, i);
            i++;
            if (i > 100)
                break;
        }
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
    bool solve(char board[Sudoku::BoardSize],
               double & elapsed_time,
               bool verbose = true) {
        if (verbose) {
            Sudoku::display_board(board, true);
        }

        jtest::StopWatch sw;
        sw.start();

        solver_.init(board);
        solver_.build(board);
        bool success = solver_.solve();

        sw.stop();
        elapsed_time = sw.getElapsedMillisec();

        if (verbose) {
            if (kSearchMode > SearchMode::OneAnswer)
                solver_.display_answers(board);
            else
                solver_.display_answer(board);
            printf("elapsed time: %0.3f ms, init_counter: %" PRIuPTR ", recur_counter: %" PRIuPTR "\n\n"
                   "num_guesses: %" PRIuPTR ", num_impossibles: %" PRIuPTR ", no_guess: %" PRIuPTR "\n"
                   "guess %% = %0.1f %%, impossible %% = %0.1f %%, no_guess %% = %0.1f %%\n\n",
                   elapsed_time, DancingLinks::get_init_counter(),
                   DancingLinks::get_search_counter(),
                   DancingLinks::get_num_guesses(),
                   DancingLinks::get_num_impossibles(),
                   DancingLinks::get_num_no_guess(),
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
