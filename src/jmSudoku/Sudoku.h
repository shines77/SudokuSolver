
#ifndef JM_SUDOKU_H
#define JM_SUDOKU_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <bitset>
#include <type_traits>

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

namespace jmSudoku {

enum SearchMode {
    OneAnswer = SEARCH_MODE_ONE_ANSWER,
    MoreThanOneAnswer = SEARCH_MODE_MORE_THAN_ONE_ANSWER,
    AllAnswers = SEARCH_MODE_ALL_ANSWERS,
    SearchModeLast
};

double calc_percent(size_t num_val, size_t num_total) {
    if (num_total != 0)
        return (num_val * 100.0) / num_total;
    else
        return 0.0;
}

template <size_t Rows, size_t Cols>
class SmallBitMatrix {
private:
    typedef std::bitset<Cols>   bitmap_type;

    size_t rows_;
    bitmap_type array_[Rows];

public:
    SmallBitMatrix() : rows_(Rows) {}
    ~SmallBitMatrix() {}

    size_t rows() const { return this->rows_; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    void setRows(size_t rows) {
        this->rows_ = rows;
    }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    size_t value(size_t row, size_t col) {
        assert(row < Rows);
        return (size_t)(this->array_[row].test(col));
    }

    void set() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
    }

    void reset() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
    }

    void flip() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
    }

    bitmap_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitmap_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Rows, size_t Cols>
class SmallBitMatrix2 {
private:
    typedef std::bitset<Cols>   bitset_type;

    bitset_type array_[Rows];

public:
    SmallBitMatrix2() = default;
    ~SmallBitMatrix2() = default;

    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    void set() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
    }

    void reset() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
    }

    void flip() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
    }

    bitset_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitset_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Depths, size_t Rows, size_t Cols>
class SmallBitMatrix3 {
private:
    typedef SmallBitMatrix2<Rows, Cols>  matrix_type;

    matrix_type matrix_[Depths];

public:
    SmallBitMatrix3() = default;
    ~SmallBitMatrix3() = default;

    size_t depths() const { return Depths; }
    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Depths; }
    size_t matrix2d_size() const { return (Rows * Cols); }
    size_t total_size() const { return (Depths * Rows * Cols); }

    bool test(size_t depth, size_t row, size_t col) {
        assert(depth < Depths);
        return this->matrix_[depth][row].test(col);
    }

    void set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
    }

    void reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
    }

    void flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
    }

    matrix_type & operator [] (size_t pos) {
        assert(pos < Depths);
        return this->matrix_[pos];
    }

    const matrix_type & operator [] (size_t pos) const {
        assert(pos < Depths);
        return this->matrix_[pos];
    }
};

template <size_t Rows, size_t Cols>
class BitMatrix2 {
private:
    typedef std::bitset<Cols>   bitset_type;

    std::vector<bitset_type> array_;

public:
    BitMatrix2() {
        this->array_.resize(Rows);
    }

    BitMatrix2(const BitMatrix2 & src) {
        this->array_.reserve(Rows);
        for (size_t row = 0; row < Rows; row++) {
            this->array_.push_back(src[row]);
        }
    }

    BitMatrix2(const SmallBitMatrix2<Rows, Cols> & src) {
        this->array_.reserve(Rows);
        for (size_t row = 0; row < Rows; row++) {
            this->array_.push_back(src[row]);
        }
    }

    ~BitMatrix2() = default;

    BitMatrix2 & operator = (const BitMatrix2 & rhs) {
        if (&rhs != this) {
            for (size_t row = 0; row < Rows; row++) {
                this->array_[row] = rhs[row];
            }
        }
    }

    BitMatrix2 & operator = (const SmallBitMatrix2<Rows, Cols> & rhs) {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row] = rhs[row];
        }
    }

    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Rows; }
    size_t total_size() const { return (Rows * Cols); }

    bool test(size_t row, size_t col) {
        assert(row < Rows);
        return this->array_[row].test(col);
    }

    void set() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].set();
        }
    }

    void reset() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].reset();
        }
    }

    void flip() {
        for (size_t row = 0; row < Rows; row++) {
            this->array_[row].flip();
        }
    }

    bitset_type & operator [] (size_t pos) {
        assert(pos < Rows);
        return this->array_[pos];
    }

    const bitset_type & operator [] (size_t pos) const {
        assert(pos < Rows);
        return this->array_[pos];
    }
};

template <size_t Depths, size_t Rows, size_t Cols>
class BitMatrix3 {
private:
    typedef BitMatrix2<Rows, Cols>  matrix_type;

    std::vector<matrix_type> matrix_;

public:
    BitMatrix3() {
        this->matrix_.resize(Depths);
    }

    BitMatrix3(const BitMatrix3 & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    BitMatrix3(const SmallBitMatrix3<Depths, Rows, Cols> & src) {
        this->matrix_.reserve(Depths);
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_.push_back(src[depth]);
        }
    }

    ~BitMatrix3() = default;

    BitMatrix3 & operator = (const BitMatrix3 & rhs) {
        if (&rhs != this) {
            for (size_t depth = 0; depth < Depths; depth++) {
                this->matrix_[depth] = rhs[depth];
            }
        }
    }

    BitMatrix3 & operator = (const SmallBitMatrix3<Depths, Rows, Cols> & rhs) {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth] = rhs[depth];
        }
    }

    size_t depths() const { return Depths; }
    size_t rows() const { return Rows; }
    size_t cols() const { return Cols; }

    size_t size() const { return Depths; }
    size_t matrix2d_size() const { return (Rows * Cols); }
    size_t total_size() const { return (Depths * Rows * Cols); }

    bool test(size_t depth, size_t row, size_t col) {
        assert(depth < Depths);
        return this->matrix_[depth][row].test(col);
    }

    void set() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].set();
        }
    }

    void reset() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].reset();
        }
    }

    void flip() {
        for (size_t depth = 0; depth < Depths; depth++) {
            this->matrix_[depth].flip();
        }
    }

    matrix_type & operator [] (size_t pos) {
        assert(pos < Depths);
        return this->matrix_[pos];
    }

    const matrix_type & operator [] (size_t pos) const {
        assert(pos < Depths);
        return this->matrix_[pos];
    }
};

template <size_t Rows, size_t Cols>
static void matrix2_copy(SmallBitMatrix2<Rows, Cols> & dest,
                         const BitMatrix2<Rows, Cols> & src)
{
    for (size_t row = 0; row < Rows; row++) {
        dest[row] = src[row];
    }
}

template <size_t Rows, size_t Cols>
static void matrix2_copy(BitMatrix2<Rows, Cols> & dest,
                         const SmallBitMatrix2<Rows, Cols> & src)
{
    for (size_t row = 0; row < Rows; row++) {
        dest[row] = src[row];
    }
}

template <size_t Depths, size_t Rows, size_t Cols>
static void matrix3_copy(SmallBitMatrix3<Depths, Rows, Cols> & dest,
                         const BitMatrix3<Depths, Rows, Cols> & src)
{
    for (size_t depth = 0; depth < Depths; depth++) {
        for (size_t row = 0; row < Rows; row++) {
            dest[depth][row] = src[depth][row];
        }
    }
}

template <size_t Depths, size_t Rows, size_t Cols>
static void matrix3_copy(BitMatrix3<Depths, Rows, Cols> & dest,
                         const SmallBitMatrix3<Depths, Rows, Cols> & src)
{
    for (size_t depth = 0; depth < Depths; depth++) {
        for (size_t row = 0; row < Rows; row++) {
            dest[depth][row] = src[depth][row];
        }
    }
}

struct Tables {
    const unsigned int div3[9] = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
    const unsigned int mod3[9] = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };

    const unsigned int roundTo3[9] = { 0, 0, 0, 3, 3, 3, 6, 6, 6 };

    const unsigned int div9[81] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 8
    };

    const unsigned int mod9[81] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8
    };

    const unsigned int palace[81] = {
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        3, 3, 3, 4, 4, 4, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8,
        6, 6, 6, 7, 7, 7, 8, 8, 8,
        6, 6, 6, 7, 7, 7, 8, 8, 8
    };

    const unsigned int palace_x_9[81] = {
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        0,   0,  0,  9,  9,  9, 18, 18, 18,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        27, 27, 27, 36, 36, 36, 45, 45, 45,
        54, 54, 54, 63, 63, 63, 72, 72, 72,
        54, 54, 54, 63, 63, 63, 72, 72, 72,
        54, 54, 54, 63, 63, 63, 72, 72, 72
    };

    const unsigned int palace_base[81] = {
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        0,   0,  0,  3,  3,  3,  6,  6,  6,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        27, 27, 27, 30, 30, 30, 33, 33, 33,
        54, 54, 54, 57, 57, 57, 60, 60, 60,
        54, 54, 54, 57, 57, 57, 60, 60, 60,
        54, 54, 54, 57, 57, 57, 60, 60, 60
    };
};

const Tables tables;

template <size_t TotalSize>
struct SudokuBoard {
    const char * board[TotalSize];
};

template <size_t nPalaceRows = 3, size_t nPalaceCols = 3,
          size_t nPalaceCountX = 3, size_t nPalaceCountY = 3,
          size_t nMinNumber = 1, size_t nMaxNumber = 9>
struct BasicSudoku {
    static const size_t PalaceRows = nPalaceRows;       // 3
    static const size_t PalaceCols = nPalaceCols;       // 3
    static const size_t PalaceCountX = nPalaceCountX;   // 3
    static const size_t PalaceCountY = nPalaceCountY;   // 3
    static const size_t MinNumber = nMinNumber;         // 1
    static const size_t MaxNumber = nMaxNumber;         // 9

    static const size_t Rows = PalaceRows * PalaceCountY;
    static const size_t Cols = PalaceCols * PalaceCountX;
    static const size_t Palaces = PalaceCountX * PalaceCountY;
    static const size_t Numbers = (MaxNumber - MinNumber) + 1;

    static const size_t PalaceSize = PalaceRows * PalaceCols;
    static const size_t BoardSize = Rows * Cols;
    static const size_t TotalSize = PalaceSize * Palaces * Numbers;

    static const size_t TotalSize2 = Rows * Cols * Numbers;

    static const size_t TotalConditions0 = 0;
    static const size_t TotalConditions1 = Rows * Cols;
    static const size_t TotalConditions2 = Rows * Numbers;
    static const size_t TotalConditions3 = Cols * Numbers;
    static const size_t TotalConditions4 = Palaces * Numbers;

    static const size_t TotalConditions01 = TotalConditions0  + TotalConditions1;
    static const size_t TotalConditions02 = TotalConditions01 + TotalConditions2;
    static const size_t TotalConditions03 = TotalConditions02 + TotalConditions3;
    static const size_t TotalConditions04 = TotalConditions03 + TotalConditions4;

    static const size_t TotalConditions =
        TotalConditions1 + TotalConditions2 + TotalConditions3 + TotalConditions4;

    static void clear_board(char board[BasicSudoku::BoardSize]) {
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                board[pos++] = '.';
            }
        }
    }

    static void clear_board(std::vector<std::vector<char>> & board) {
        for (size_t row = 0; row < board.size(); row++) {
            std::vector<char> & line = board[row];
            for (size_t col = 0; col < line.size(); col++) {
                line[col] = '.';
            }
        }
    }

    static void display_board(char board[BasicSudoku::BoardSize],
                              bool is_input = false,
                              int idx = -1) {
        int filled = 0;
        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                if (board[pos] != '.')
                    filled++;
            }
        }

        if (is_input) {
            printf("The input is: (filled = %d)\n", filled);
        }
        else {
            if (idx == -1)
                printf("The answer is:\n");
            else
                printf("The answer # %d is:\n", idx + 1);
        }
        printf("\n");
        // printf("  ------- ------- -------\n");
        printf(" ");
        for (size_t countX = 0; countX < PalaceCountX; countX++) {
            printf(" -------");
        }
        printf("\n");
        pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            printf(" | ");
            for (size_t col = 0; col < Cols; col++) {
                char val = board[pos++];
                if (val == ' ' || val == '0' || val == '-')
                    printf(". ");
                else
                    printf("%c ", val);
                if ((col % PalaceCols) == (PalaceCols - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % PalaceRows) == (PalaceRows - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < PalaceCountX; countX++) {
                    printf(" -------");
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    static void display_board(const std::vector<std::vector<char>> & board,
                              bool is_input = false,
                              int idx = -1) {
        int filled = 0;
        for (size_t row = 0; row < board.size(); row++) {
            const std::vector<char> & line = board[row];
            for (size_t col = 0; col < line.size(); col++) {
                if (line[col] != '.')
                    filled++;
            }
        }

        if (is_input) {
            printf("The input is: (filled = %d)\n", filled);
        }
        else {
            if (idx == -1)
                printf("The answer is:\n");
            else
                printf("The answer # %d is:\n", idx + 1);
        }
        printf("\n");
        // printf("  ------- ------- -------\n");
        printf(" ");
        for (size_t countX = 0; countX < PalaceCountX; countX++) {
            printf(" -------");
        }
        printf("\n");
        for (size_t row = 0; row < Rows; row++) {
            assert(board.size() >= Rows);
            printf(" | ");
            for (size_t col = 0; col < Cols; col++) {
                assert(board[row].size() >= Cols);
                char val = board[row][col];
                if (val == ' ' || val == '0' || val == '-')
                    printf(". ");
                else
                    printf("%c ", val);
                if ((col % PalaceCols) == (PalaceCols - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % PalaceRows) == (PalaceRows - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < PalaceCountX; countX++) {
                    printf(" -------");
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    static void display_boards(std::vector<SudokuBoard<BasicSudoku::TotalSize>> & boards) {
        printf("Total answers: %d\n\n", (int)boards.size());
        int i = 0;
        for (auto board : boards) {
            BasicSudoku::display_board(board, false, i);
            i++;
        }
    }

    static void display_boards(std::vector<std::vector<std::vector<char>>> & boards) {
        printf("Total answers: %d\n\n", (int)boards.size());
        int i = 0;
        for (auto board : boards) {
            BasicSudoku::display_board(board, false, i);
            i++;
        }
    }
};

// Standard sudoku definition
typedef BasicSudoku<3, 3, 3, 3, 1, 9> Sudoku;

} // namespace jmSudoku

#endif // JM_SUDOKU_H
