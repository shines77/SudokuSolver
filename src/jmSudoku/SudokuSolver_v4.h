
#ifndef JM_SUDOKU_SOLVER_V4_H
#define JM_SUDOKU_SOLVER_V4_H

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
#include <cstring>      // For std::memset(), std::memcpy()
#include <vector>
#include <bitset>
#include <array>        // For std::array<T, Size>

#if defined(_MSC_VER)
#include <emmintrin.h>      // For SSE 2
#include <tmmintrin.h>      // For SSE 3
#include <smmintrin.h>      // For SSE 4.1
#else
#include <x86intrin.h>      // For SSE 4.1
#endif // _MSC_VER

#include "Sudoku.h"
#include "StopWatch.h"
#include "BitUtils.h"
#include "BitSet.h"
#include "PackedBitSet.h"
#include "BitMatrix.h"
#include "BitVec.h"

/************************************************

#define SEARCH_MODE_ONE_ANSWER              0
#define SEARCH_MODE_MORE_THAN_ONE_ANSWER    1
#define SEARCH_MODE_ALL_ANSWERS             2

************************************************/

#define V4_SEARCH_MODE          SEARCH_MODE_ONE_ANSWER

#define V4_LITERAL_ORDER_MODE   0

namespace jmSudoku {
namespace v4 {

static const size_t kSearchMode = V3E_SEARCH_MODE;

template <typename SudokuTy>
class Solver {
public:
    typedef SudokuTy                            sudoku_type;
    typedef Solver<SudokuTy>                    solver_type;
    typedef typename SudokuTy::board_type       Board;
    typedef typename SudokuTy::NeighborCells    NeighborCells;
    typedef typename SudokuTy::CellInfo         CellInfo;
    typedef typename SudokuTy::BoxesInfo        BoxesInfo;

    typedef typename SudokuTy::BitMask          BitMask;
    typedef typename SudokuTy::BitMaskTable     BitMaskTable;

    static const size_t kAlignment = SudokuTy::kAlignment;
    static const size_t BoxCellsX = SudokuTy::BoxCellsX;      // 3
    static const size_t BoxCellsY = SudokuTy::BoxCellsY;      // 3
    static const size_t BoxCountX = SudokuTy::BoxCountX;      // 3
    static const size_t BoxCountY = SudokuTy::BoxCountY;      // 3
    static const size_t MinNumber = SudokuTy::MinNumber;      // 1
    static const size_t MaxNumber = SudokuTy::MaxNumber;      // 9

    static const size_t Rows = SudokuTy::Rows;
    static const size_t Cols = SudokuTy::Cols;
    static const size_t Boxes = SudokuTy::Boxes;
    static const size_t BoxSize = SudokuTy::BoxSize;
    static const size_t Numbers = SudokuTy::Numbers;

    static const size_t BoardSize = SudokuTy::BoardSize;
    static const size_t TotalSize = SudokuTy::TotalSize;
    static const size_t Neighbors = SudokuTy::Neighbors;

    static const size_t Rows16 = AlignedTo<Rows, 16>::value;
    static const size_t Cols16 = AlignedTo<Cols, 16>::value;
    static const size_t Numbers16 = AlignedTo<Numbers, 16>::value;
    static const size_t Boxes16 = AlignedTo<Boxes, 16>::value;
    static const size_t BoxSize16 = AlignedTo<BoxSize, 16>::value;
    static const size_t BoardSize16 = Boxes16 * BoxSize16;

    static const size_t Rows32 = Rows16 * 2;
    static const size_t Cols32 = Cols16 * 2;
    static const size_t Numbers32 = Numbers16 * 2;
    static const size_t Boxes32 = Boxes16 * 2;
    static const size_t BoxSize32 = BoxSize16 * 2;
    static const size_t BoardSize32 = Boxes32 * BoxSize32;

    static const size_t TotalCellLiterals = Boxes16 * BoxSize16;
    static const size_t TotalRowLiterals = Rows16 * Numbers16;
    static const size_t TotalColLiterals = Cols16 * Numbers16;
    static const size_t TotalBoxLiterals = Boxes16 * Numbers16;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

#if (V4_LITERAL_ORDER_MODE == 0)
    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t RowLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t BoxLiteralFirst  = ColLiteralFirst + TotalColLiterals;
    static const size_t LiteralLast      = BoxLiteralFirst + TotalBoxLiterals;

    static const size_t CellLiteralLast  = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = LiteralLast;
#else
    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t BoxLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t RowLiteralFirst  = BoxLiteralFirst + TotalBoxLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t LiteralLast      = ColLiteralFirst + TotalColLiterals;

    static const size_t CellLiteralLast  = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = LiteralLast;
#endif // (V4_LITERAL_ORDER_MODE == 0)

    static const size_t kAllRowsBits = SudokuTy::kAllRowsBit;
    static const size_t kAllColsBits = SudokuTy::kAllColsBit;
    static const size_t kAllBoxesBits = SudokuTy::kAllBoxesBit;
    static const size_t kAllBoxSizeBits = SudokuTy::kAllBoxSizeBits;
    static const size_t kAllNumbersBits = SudokuTy::kAllNumbersBits;

    static const bool kAllDimIsSame = SudokuTy::kAllDimIsSame;

    static const int kLiteralCntThreshold = 0;
    static const uint32_t kLiteralCntThreshold2 = 0;

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_failed_return;

private:
#if (V3_LITERAL_ORDER_MODE == 0)
    enum LiteralType {
        CellNums,
        RowNums,
        ColNums,
        BoxNums,
        MaxLiteralType
    };
#else
    enum LiteralType {
        CellNums,
        BoxNums,
        RowNums,
        ColNums,
        MaxLiteralType
    };
#endif // (V3_LITERAL_ORDER_MODE == 0)

private:

#pragma pack(push, 1)

    struct State {
        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
        alignas(32) PackedBitSet3D<Numbers, Rows16, Cols16>       num_row_cols;     // [num][row][col]
        alignas(32) PackedBitSet3D<Numbers, Cols16, Rows16>       num_col_rows;     // [num][col][row]
        alignas(32) PackedBitSet3D<Numbers, Boxes16, BoxSize16>   num_box_cells;    // [num][box][cell]
    };

#pragma pack(pop)

    State   state_;

    size_t  empties_;

    std::vector<Board>  answers_;

    static bool is_mask_inited;
    static std::vector<neighbor_boxes_t> neighbor_boxes;

    static PackedBitSet2D<BoardSize, Rows16 * Cols16>     neighbor_cells_mask;
    static PackedBitSet2D<BoardSize, Boxes16 * BoxSize16> neighbor_boxes_mask;

    static PackedBitSet3D<Boxes, BoxSize16, Numbers16>    box_cell_neighbors_mask[BoardSize][Numbers];
    static PackedBitSet3D<BoardSize, Rows16, Cols16>      row_neighbors_mask;
    static PackedBitSet3D<BoardSize, Cols16, Rows16>      col_neighbors_mask;
    static PackedBitSet3D<BoardSize, Boxes16, BoxSize16>  box_num_neighbors_mask;

public:
    Solver() : empties_(0) {
        if (!is_mask_inited) {
            init_mask();
            is_mask_inited = true;
        }
    }
    ~Solver() {}

    static size_t get_num_guesses() { return solver_type::num_guesses; }
    static size_t get_num_unique_candidate() { return solver_type::num_unique_candidate; }
    static size_t get_num_failed_return() { return solver_type::num_failed_return; }

    static size_t get_total_search_counter() {
        return (solver_type::num_guesses + solver_type::num_unique_candidate + solver_type::num_failed_return);
    }

    static double get_guess_percent() {
        return calc_percent(solver_type::num_guesses, solver_type::get_total_search_counter());
    }

    static double get_failed_return_percent() {
        return calc_percent(solver_type::num_failed_return, solver_type::get_total_search_counter());
    }

    static double get_unique_candidate_percent() {
        return calc_percent(solver_type::num_unique_candidate, solver_type::get_total_search_counter());
    }

private:
    size_t calc_empties(Board & board) {
        size_t empties = 0;
        for (size_t pos = 0; pos < BoardSize; pos++) {
            unsigned char val = board.cells[pos];
            if (val == '.') {
                empties++;
            }
        }
        return empties;
    }

    static size_t make_neighbor_cells_masklist(size_t fill_pos,
                                               size_t row, size_t col) {
        PackedBitSet<BoardSize16> & cells_mask = neighbor_cells_mask[fill_pos];
        PackedBitSet<BoardSize16> & boxes_mask = neighbor_boxes_mask[fill_pos];

        PackedBitSet2D<Rows16, Cols16> & rows_mask        = row_neighbors_mask[fill_pos];
        PackedBitSet2D<Cols16, Rows16> & cols_mask        = col_neighbors_mask[fill_pos];
        PackedBitSet2D<Boxes16, BoxSize16> & box_num_mask = box_num_neighbors_mask[fill_pos];

        const CellInfo * pCellInfo = SudokuTy::cell_info;
        size_t box, cell;

        rows_mask[row].set(col);
        cols_mask[col].set(row);

        const CellInfo & cellInfo2 = pCellInfo[fill_pos];
        box = cellInfo2.box;
        cell = cellInfo2.cell;
        box_num_mask[box].set(cell);

        size_t index = 0;
        size_t pos_y = row * Cols;
        for (size_t x = 0; x < Cols; x++) {
            if (x != col) {
                size_t pos = pos_y + x;
                size_t pos16 = row * Cols16 + x;
                cells_mask.set(pos16);

                const CellInfo & cellInfo = pCellInfo[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                size_t box_pos16 = box * BoxSize16 + cell;
                boxes_mask.set(box_pos16);

                rows_mask[row].set(x);
                cols_mask[x].set(row);
                box_num_mask[box].set(cell);

                for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                    PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_cell_mask = box_cell_neighbors_mask[fill_pos][num];
                    box_cell_mask[box][cell].set(num);
                }

                index++;
            }
        }

        size_t pos_x = col;
        for (size_t y = 0; y < Rows; y++) {
            if (y != row) {
                size_t pos = y * Cols + pos_x;
                size_t pos16 = y * Cols16 + pos_x;
                cells_mask.set(pos16);

                const CellInfo & cellInfo = pCellInfo[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                size_t box_pos16 = box * BoxSize16 + cell;
                boxes_mask.set(box_pos16);

                rows_mask[y].set(col);
                cols_mask[col].set(y);
                box_num_mask[box].set(cell);

                for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                    PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_cell_mask = box_cell_neighbors_mask[fill_pos][num];
                    box_cell_mask[box][cell].set(num);
                }

                index++;
            }
        }

        size_t box_x = col / BoxCellsX;
        size_t box_y = row / BoxCellsY;
        size_t box_base = (box_y * BoxCellsY) * Cols + box_x * BoxCellsX;
        size_t cell_x = col % BoxCellsX;
        size_t cell_y = row % BoxCellsY;
        size_t pos = box_base;
        for (size_t y = 0; y < BoxCellsY; y++) {
            if (y == cell_y) {
                pos += Cols;
            }
            else {
                for (size_t x = 0; x < BoxCellsX; x++) {
                    if (x != cell_x) {
                        assert(pos != fill_pos);
                        const CellInfo & cellInfo = pCellInfo[pos];
                        box = cellInfo.box;
                        cell = cellInfo.cell;
                        row = cellInfo.row;
                        col = cellInfo.col;

                        size_t pos16 = row * Cols16 + col;
                        cells_mask.set(pos16);

                        size_t box_pos16 = box * BoxSize16 + cell;
                        boxes_mask.set(box_pos16);

                        rows_mask[row].set(col);
                        cols_mask[col].set(row);
                        box_num_mask[box].set(cell);

                        for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                            PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_cell_mask = box_cell_neighbors_mask[fill_pos][num];
                            box_cell_mask[box][cell].set(num);
                        }

                        index++;
                    }
                    pos++;
                }
                pos += (Cols - BoxCellsX);
            }
        }

        assert(index == Neighbors);
        return index;
    }

    static void init_neighbor_cells_mask() {
        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_neighbor_cells_masklist(fill_pos, row, col);
                fill_pos++;
            }
        }
    }

    static void init_neighbor_boxes() {
        neighbor_boxes.reserve(Boxes);
        for (size_t box_y = 0; box_y < BoxCellsY; box_y++) {
            size_t box_y_base = box_y * BoxCellsX;
            for (size_t box_x = 0; box_x < BoxCellsX; box_x++) {
                size_t box = box_y_base + box_x;
                size_t index = 0;
                neighbor_boxes_t neighborBoxes;
                neighborBoxes.boxes[index++] = box;
                for (size_t box_i = 0; box_i < BoxCellsX; box_i++) {
                    if (box_i != box_x) {
                        neighborBoxes.boxes[index++] = box_y * BoxCellsX + box_i;
                    }
                }
                for (size_t box_j = 0; box_j < BoxCellsY; box_j++) {
                    if (box_j != box_y) {
                        neighborBoxes.boxes[index++] = box_j * BoxCellsX + box_x;
                    }
                }
                assert(index == neighborBoxes.boxes_count());

                std::sort(&neighborBoxes.boxes[1], &neighborBoxes.boxes[neighborBoxes.boxes_count()]);
                neighbor_boxes.push_back(neighborBoxes);
            }
        }
    }

    static void init_mask() {
        neighbor_cells_mask.reset();

        for (size_t pos = 0; pos < BoardSize; pos++) {
            for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                box_cell_neighbors_mask[pos][num].reset();
            }
        }
        row_neighbors_mask.reset();
        col_neighbors_mask.reset();
        box_num_neighbors_mask.reset();

        init_neighbor_boxes();
        init_neighbor_cells_mask();

        // Flip all mask bits
        for (size_t pos = 0; pos < BoardSize; pos++) {
            for (size_t num = MinNumber - 1; num < MaxNumber; num++) {
                box_cell_neighbors_mask[pos][num].flip();
            }
        }
        row_neighbors_mask.flip();
        col_neighbors_mask.flip();
        box_num_neighbors_mask.flip();
    }


    void init_board(Board & board) {
        this->state_.box_cell_nums.fill(kAllNumbersBit);
        this->state_.row_num_cols.fill(kAllColsBit);
        this->state_.col_num_rows.fill(kAllRowsBit);
        this->state_.box_num_cells.fill(kAllBoxSizeBit);

        num_guesses = 0;
        num_unique_candidate = 0;
        num_failed_return = 0;
        if (kSearchMode > SEARCH_MODE_ONE_ANSWER) {
            this->answers_.clear();
        }

        size_t empties = calc_empties(board);
        this->empties_ = empties;

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            size_t cell_y = (row % BoxCellsY) * BoxCellsX;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board.cells[pos];
                if (val != '.') {
                    size_t box_x = col / BoxCellsX;
                    size_t box = box_y + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell = cell_y + cell_x;
                    size_t num = val - '1';
#if V3E_ENABLE_OLD_ALGORITHM
                    this->old_doFillNum(pos, row, col, box, cell, num);
                    this->old_updateNeighborCellsEffect(pos, num);
#endif
                    this->doFillNum(pos, row, col, box, cell, num);
                    this->updateNeighborCellsEffect(pos, box, num);
                }
                pos++;
            }
        }

        uint32_t min_literal_index;
        uint32_t min_literal_size = this->count_all_literal_size(min_literal_index);
        this->count_.min_literal_size = min_literal_size;
        this->count_.min_literal_index = min_literal_index;

#if V3E_ENABLE_OLD_ALGORITHM
        bool is_correct = verify_bitboard_state();
        assert(is_correct);

        bool size_is_correct = verify_literal_size();
        assert(size_is_correct);
#endif
    }
};

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_guesses = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_unique_candidate = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_failed_return = 0;

} // namespace v4
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V4_H
