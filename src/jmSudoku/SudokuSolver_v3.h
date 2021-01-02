
#ifndef JM_SUDOKU_SOLVER_V3_H
#define JM_SUDOKU_SOLVER_V3_H

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

#define V3_SEARCH_MODE          SEARCH_MODE_ONE_ANSWER

#define V3_LITERAL_ORDER_MODE   0

#ifdef _MSC_VER
#define V3_USE_STD_BITSET       0
#else
#define V3_USE_STD_BITSET       0
#endif

#ifdef NDEBUG
#define V3_SAVE_COUNT_SIZE      1
#else
#define V3_SAVE_COUNT_SIZE      1
#endif

#define V3_ENABLE_OLD_ALGORITHM     0

namespace jmSudoku {
namespace v3 {

static const size_t kSearchMode = V3_SEARCH_MODE;

#if 0
static const NeighborBoxes<3, 3> neighbor_boxes[9] = {
    // Box # 0
    { { 1, 2, 3, 6 } },
    // Box # 1
    { { 0, 2, 4, 7 } },
    // Box # 2
    { { 0, 1, 5, 8 } },
    // Box # 3
    { { 4, 5, 0, 6 } },
    // Box # 4
    { { 3, 5, 1, 7 } },
    // Box # 5
    { { 3, 4, 2, 8 } },
    // Box # 6
    { { 7, 8, 0, 3 } },
    // Box # 7
    { { 6, 8, 1, 4 } },
    // Box # 8
    { { 6, 7, 2, 5 } }
};
#endif

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

#if (V3_LITERAL_ORDER_MODE == 0)
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
#endif // (V3_LITERAL_ORDER_MODE == 0)

    static const size_t kAllRowsBit = SudokuTy::kAllRowsBit;
    static const size_t kAllColsBit = SudokuTy::kAllColsBit;
    static const size_t kAllBoxesBit = SudokuTy::kAllBoxesBit;
    static const size_t kAllBoxSizeBit = SudokuTy::kAllBoxSizeBit;
    static const size_t kAllNumbersBit = SudokuTy::kAllNumbersBit;

    static const bool kAllDimIsSame = SudokuTy::kAllDimIsSame;

    static const int kLiteralCntThreshold = 0;

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

#pragma pack(push, 1)

    union literal_info_t {
        struct {
            uint8_t count;
            uint8_t enable;
        };
        uint16_t value;
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct NeighborBoxes {
        static const size_t kBoxesCount = (nBoxCountX - 1) + (nBoxCountY - 1) + 1;
        size_t boxes_count() const { return kBoxesCount; }

        size_t boxes[kBoxesCount];
    };

    typedef NeighborBoxes<BoxCountX, BoxCountY> neighbor_boxes_t;

    struct State {
        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
        alignas(32) PackedBitSet3D<Numbers, Rows16, Cols16>       row_num_cols;     // [num][row][col]
        alignas(32) PackedBitSet3D<Numbers, Cols16, Rows16>       col_num_rows;     // [num][col][row]
        alignas(32) PackedBitSet3D<Numbers, Boxes16, BoxSize16>   box_num_cells;    // [num][box][cell]
    };

    struct RecoverState {
        static const size_t kBoxesTotal = neighbor_boxes_t::kBoxesCount;
        alignas(32) PackedBitSet2D<BoxSize16, Numbers16>          boxes[kBoxesTotal];   // [cell][num]
        alignas(32) PackedBitSet2D<Rows16, Cols16>                row_cols;             // [row][col]
        alignas(32) PackedBitSet2D<Cols16, Rows16>                col_rows;             // [col][row]
        alignas(32) PackedBitSet2D<Boxes16, BoxSize16>            box_cells;            // [box][cell]

        struct Counts {
            alignas(32) uint16_t box_cells[Boxes16];
            alignas(32) uint16_t row_nums[Numbers16];
            alignas(32) uint16_t col_nums[Numbers16];
            alignas(32) uint16_t box_nums[Numbers16];
        } counts;

        struct Indexs {
            alignas(32) uint16_t box_cells[Boxes16];
            alignas(32) uint16_t row_nums[Numbers16];
            alignas(32) uint16_t col_nums[Numbers16];
            alignas(32) uint16_t box_nums[Numbers16];
        } indexs;
    };

    struct Count {
        struct Sizes {
            alignas(32) uint16_t box_cells[Boxes16 * BoxSize16];
            alignas(32) uint16_t row_nums[Numbers16 * Rows16];
            alignas(32) uint16_t col_nums[Numbers16 * Cols16];
            alignas(32) uint16_t box_nums[Numbers16 * Boxes16];
        } sizes;

        struct Enabled {
            alignas(32) uint16_t box_cells[Boxes16 * BoxSize16];
            alignas(32) uint16_t row_nums[Numbers16 * Rows16];
            alignas(32) uint16_t col_nums[Numbers16 * Cols16];
            alignas(32) uint16_t box_nums[Numbers16 * Boxes16];
        } enabled;

        struct Counts {
            alignas(32) uint16_t box_cells[Boxes16];
            alignas(32) uint16_t row_nums[Numbers16];
            alignas(32) uint16_t col_nums[Numbers16];
            alignas(32) uint16_t box_nums[Numbers16];
        } counts;

        struct Indexs {
            alignas(32) uint16_t box_cells[Boxes16];
            alignas(32) uint16_t row_nums[Numbers16];
            alignas(32) uint16_t col_nums[Numbers16];
            alignas(32) uint16_t box_nums[Numbers16];
        } indexs;

        struct Total {
            alignas(32) uint16_t min_literal_size[16];
            alignas(32) uint16_t min_literal_index[16];
        } total;

        uint32_t min_literal_size;
        uint32_t min_literal_index;
    };

#pragma pack(pop)

    typedef PackedBitSet<Numbers16>     bitset_type;

    alignas(32) PackedBitSet2D<Numbers, Boxes16 * BoxSize16>  num_cells_;       // [num][box * BoxSize16 + cell]

    alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums_;   // [box][cell][num]
    alignas(32) PackedBitSet3D<Numbers, Rows16, Cols16>       row_num_cols_;    // [num][row][col]
    alignas(32) PackedBitSet3D<Numbers, Cols16, Rows16>       col_num_rows_;    // [num][col][row]
    alignas(32) PackedBitSet3D<Numbers, Boxes16, BoxSize16>   box_num_cells_;   // [num][box][cell]

    State   state_;
    Count   count_;

#if defined(__SSE4_1__)
    alignas(16) literal_info_t literal_info_[TotalLiterals];
#else
    alignas(16) uint8_t literal_count_[TotalLiterals];
    alignas(16) uint8_t literal_enable_[TotalLiterals];
#endif

    size_t empties_;

    std::vector<Board>  answers_;

    static bool is_mask_inited;

    static std::vector<neighbor_boxes_t> neighbor_boxes;

    static alignas(32) PackedBitSet2D<BoardSize, Rows16 * Cols16>     neighbor_cells_mask;
    static alignas(32) PackedBitSet2D<BoardSize, Boxes16 * BoxSize16> neighbor_boxes_mask;

    static alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>    box_cell_neighbors_mask[BoardSize][Numbers];
    static alignas(32) PackedBitSet3D<BoardSize, Rows16, Cols16>      row_neighbors_mask;
    static alignas(32) PackedBitSet3D<BoardSize, Cols16, Rows16>      col_neighbors_mask;
    static alignas(32) PackedBitSet3D<BoardSize, Boxes16, BoxSize16>  box_num_neighbors_mask;

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
                neighborBoxes.boxes[index++] = box;
                assert(index == neighborBoxes.boxes_count());

                std::sort(&neighborBoxes.boxes[0], &neighborBoxes.boxes[neighborBoxes.boxes_count()]);
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
#if V3_ENABLE_OLD_ALGORITHM
        init_literal_info();
#endif
        _init_literal_info();

#if V3_ENABLE_OLD_ALGORITHM
        size_t kBoxSize64 = kAllBoxSizeBit | (kAllBoxSizeBit << 16U) | (kAllBoxSizeBit << 32U) | (kAllBoxSizeBit << 48U);
        this->num_cells_.fill(kBoxSize64);

        this->box_cell_nums_.fill(kAllNumbersBit);
        this->row_num_cols_.fill(kAllColsBit);
        this->col_num_rows_.fill(kAllRowsBit);
        this->box_num_cells_.fill(kAllBoxSizeBit);
#endif
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
#if V3_ENABLE_OLD_ALGORITHM
                    this->doFillNum(pos, row, col, box, cell, num);
                    this->updateNeighborCellsEffect(pos, num);
#endif
                    this->_doFillNum(pos, row, col, box, cell, num);
                    this->_updateNeighborCellsEffect(pos, box, num);
                }
                pos++;
            }
        }

        uint32_t min_literal_index;
        uint32_t min_literal_size = this->count_all_literal_size(min_literal_index);
        this->count_.min_literal_size = min_literal_size;
        this->count_.min_literal_index = min_literal_index;

        bool is_correct = verify_bitboard_state();
        assert(is_correct);

        bool size_is_correct = verify_literal_size();
        assert(size_is_correct);
    }

    static const size_t kLiteralStep = sizeof(size_t) / sizeof(literal_info_t);

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
    static const size_t kInitCellLiteral = Numbers | (Numbers << 16U) | (Numbers << 32U) | (Numbers << 48U);
    static const size_t kInitRowLiteral  = Cols | (Cols << 16U) | (Cols << 32U) | (Cols << 48U);
    static const size_t kInitColLiteral  = Rows | (Rows << 16U) | (Rows << 32U) | (Rows << 48U);
    static const size_t kInitBoxLiteral  = BoxSize | (BoxSize << 16U) | (BoxSize << 32U) | (BoxSize << 48U);
    static const size_t kInitLiteral     = kInitCellLiteral;
#else
    static const size_t kInitCellLiteral = Numbers | (Numbers << 16U);
    static const size_t kInitRowLiteral  = Cols | (Cols << 16U);
    static const size_t kInitColLiteral  = Rows | (Rows << 16U);
    static const size_t kInitBoxLiteral  = BoxSize | (BoxSize << 16U);
    static const size_t kInitLiteral     = kInitCellLiteral;
#endif

#if defined(__SSE4_1__)
    inline void init_literal_info() {
        if (kAllDimIsSame)
            init_literal_info_is_same();
        else
            init_literal_info_not_same();
    }

    inline void init_literal_info_is_same() {
        static const size_t kLiteralLimit = (LiteralLast / kLiteralStep) * kLiteralStep;

        // Cell-Literal, Row-Literal, Col-Literal, Box-Literal
        size_t * pinfo = (size_t *)(&this->literal_info_[0]);
        assert((uintptr_t(pinfo) & 0x0FU) == 0);

        size_t * pinfo_limit = (size_t *)(&this->literal_info_[kLiteralLimit]);
        while (pinfo < pinfo_limit) {
            *pinfo = kInitLiteral;
            pinfo++;
        }
        for (size_t i = kLiteralLimit; i < LiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }
    }

    inline void init_literal_info_not_same() {
        size_t * pinfo_first = (size_t *)(&this->literal_info_[0]);
        assert((uintptr_t(pinfo_first) & 0x0FU) == 0);

        size_t i, literalFront, literalLast;

        // Cell-Literal
        literalLast = (CellLiteralLast / kLiteralStep) * kLiteralStep;
        for (i = CellLiteralFirst; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitCellLiteral;
        }
        for (; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }

        // Row-Literal
        literalFront = RowLiteralFirst + (RowLiteralFirst % kLiteralStep);
        for (i = RowLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }
        literalLast = (RowLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitRowLiteral;
        }
        for (; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }

        // Col-Literal
        literalFront = ColLiteralFirst + (ColLiteralFirst % kLiteralStep);
        for (i = ColLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }
        literalLast = (ColLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitColLiteral;
        }
        for (; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }

        // Box-Literal
        literalFront = BoxLiteralFirst + (BoxLiteralFirst % kLiteralStep);
        for (i = BoxLiteralFirst; i < literalFront; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
        literalLast = (BoxLiteralLast / kLiteralStep) * kLiteralStep;
        for (; i < literalLast; i += kLiteralStep) {
            size_t * pinfo = (size_t *)&this->literal_info_[i];
            *pinfo = kInitBoxLiteral;
        }
        for (; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
    }

    inline void init_literal_info_normal() {
#if 0
        std::memset((void *)&this->literal_info_[0], 0, sizeof(this->literal_info_));
#endif
        for (size_t i = CellLiteralFirst; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Numbers);
        }
        for (size_t i = RowLiteralFirst; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Cols);
        }
        for (size_t i = ColLiteralFirst; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint16_t)Rows);
        }
        for (size_t i = BoxLiteralFirst; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint16_t)BoxSize);
        }
    }

    inline void init_literal_info(size_t literal, uint16_t value) {
        this->literal_info_[literal].value = value;
    }

    inline void enable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_info_[literal].enable = 0xFF;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_info_[literal].count;
    }

    inline uint8_t get_literal_enable(size_t literal) {
        return this->literal_info_[literal].enable;
    }

    inline void set_literal_cnt(size_t literal, uint8_t count) {
        this->literal_info_[literal].count = count;
    }

    inline void set_literal_enable(size_t literal, uint8_t enable) {
        this->literal_info_[literal].enable = enable;
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_info_[literal].count++;
        assert(this->literal_info_[literal].count <= Numbers);
    }

    inline void dec_literal_cnt(size_t literal) {
        assert(this->literal_info_[literal].count > 0);
        this->literal_info_[literal].count--;
    }

#else // !__SSE4_1__

    inline void init_literal_info() {
        if (kAllDimIsSame)
            init_literal_info_is_same();
        else
            init_literal_info_not_same();
    }

    inline void init_literal_info_is_same() {
        std::memset((void *)&this->literal_count_[0],  (int)kInitLiteral, sizeof(this->literal_count_));
        std::memset((void *)&this->literal_enable_[0], 0,                 sizeof(this->literal_enable_));
    }

    inline void init_literal_info_not_same() {
        for (size_t i = CellLiteralFirst; i < CellLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Numbers);
        }
        for (size_t i = RowLiteralFirst; i < RowLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Cols);
        }
        for (size_t i = ColLiteralFirst; i < ColLiteralLast; i++) {
            init_literal_info(i, (uint8_t)Rows);
        }
        for (size_t i = BoxLiteralFirst; i < BoxLiteralLast; i++) {
            init_literal_info(i, (uint8_t)BoardSize);
        }

        std::memset((void *)&this->literal_enable_[0], 0, sizeof(this->literal_enable_));
    }

    inline void init_literal_info(size_t literal, uint8_t count) {
        this->literal_count_[literal] = count;
    }

    inline void enable_literal(size_t literal) {
        this->literal_enable_[literal] = 0x00;
    }

    inline void disable_literal(size_t literal) {
        this->literal_enable_[literal] = 0xF0;
    }

    inline uint8_t get_literal_cnt(size_t literal) {
        return this->literal_count_[literal];
    }

    inline uint8_t get_literal_enable(size_t literal) {
        return this->literal_enable_[literal];
    }

    inline void set_literal_cnt(size_t literal, uint8_t count) {
        this->literal_count_[literal] = count;
    }

    inline void set_literal_enable(size_t literal, uint8_t enable) {
        this->literal_enable_[literal] = enable;
    }

    inline void inc_literal_cnt(size_t literal) {
        this->literal_count_[literal]++;
        assert(this->literal_count_[literal] <= Numbers);
    }

    inline void dec_literal_cnt(size_t literal) {
        assert(this->literal_count_[literal] > 0);
        this->literal_count_[literal]--;
    }

#endif // __SSE4_1__

    // enable_xxxx_literal()
    inline void enable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_row_literal(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_col_literal(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->enable_literal(literal);
    }

    inline void enable_box_literal(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        this->enable_literal(literal);
    }

    // disable_xxxx_literal()
    inline void disable_cell_literal(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_row_literal(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_col_literal(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->disable_literal(literal);
    }

    inline void disable_box_literal(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        this->disable_literal(literal);
    }

    // get_xxxx_literal_enable()
    inline uint8_t get_cell_literal_enable(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        return this->get_literal_enable(literal);
    }

    inline uint8_t get_row_literal_enable(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        return this->get_literal_enable(literal);
    }

    inline uint8_t get_col_literal_enable(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        return this->get_literal_enable(literal);
    }

    inline uint8_t get_box_literal_enable(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        return this->get_literal_enable(literal);
    }

    // get_xxxx_literal_cnt()
    inline uint8_t get_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        return this->get_literal_cnt(literal);
    }

    inline uint8_t get_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        return this->get_literal_cnt(literal);
    }

    // set_xxxx_literal_cnt()
    inline void set_cell_literal_cnt(size_t pos, uint8_t count) {
        size_t literal = CellLiteralFirst + pos;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t index, uint8_t count) {
        size_t literal = RowLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_row_literal_cnt(size_t num, size_t row, uint8_t count) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t index, uint8_t count) {
        size_t literal = ColLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_col_literal_cnt(size_t num, size_t col, uint8_t count) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t index, uint8_t count) {
        size_t literal = BoxLiteralFirst + index;
        this->set_literal_cnt(literal, count);
    }

    inline void set_box_literal_cnt(size_t num, size_t box, uint8_t count) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        this->set_literal_cnt(literal, count);
    }

    // inc_xxxx_literal_cnt()
    inline void inc_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->inc_literal_cnt(literal);
    }

    inline void inc_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        this->inc_literal_cnt(literal);
    }

    // dec_xxxx_literal_cnt()
    inline void dec_cell_literal_cnt(size_t pos) {
        size_t literal = CellLiteralFirst + pos;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t index) {
        size_t literal = RowLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_row_literal_cnt(size_t num, size_t row) {
        size_t literal = RowLiteralFirst + num * Rows16 + row;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t index) {
        size_t literal = ColLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_col_literal_cnt(size_t num, size_t col) {
        size_t literal = ColLiteralFirst + num * Cols16 + col;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t index) {
        size_t literal = BoxLiteralFirst + index;
        this->dec_literal_cnt(literal);
    }

    inline void dec_box_literal_cnt(size_t num, size_t box) {
        size_t literal = BoxLiteralFirst + num * Boxes16 + box;
        this->dec_literal_cnt(literal);
    }

#if defined(__SSE4_1__)
    int get_min_literal_normal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_info_[i].enable == 0) {
                int literal_cnt = literal_info_[i].count;
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return i;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

    int get_min_literal(int & out_min_literal_cnt) {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        int index_base = 0;

        const char * pinfo     = (const char *)&literal_info_[0];
        const char * pinfo_end = (const char *)&literal_info_[TotalLiterals];
        while ((pinfo_end - pinfo) >= 64) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pinfo + 16));
            __m128i xmm2 = _mm_load_si128((const __m128i *)(pinfo + 32));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(pinfo + 48));

            __m128i xmm4 = _mm_minpos_epu16(xmm0);      // SSE 4.1
            __m128i xmm5 = _mm_minpos_epu16(xmm1);      // SSE 4.1
            __m128i xmm6 = _mm_minpos_epu16(xmm2);      // SSE 4.1
            __m128i xmm7 = _mm_minpos_epu16(xmm3);      // SSE 4.1

            __m128i xmm5_ = _mm_slli_epi64(xmm5, 32);
            __m128i xmm7_ = _mm_slli_epi64(xmm7, 32);
            __m128i xmm4_5  = _mm_blend_epi16(xmm4, xmm5_, 0b00001100); // SSE 4.1
            __m128i xmm6_7  = _mm_blend_epi16(xmm6, xmm7_, 0b00001100); // SSE 4.1
            __m128i xmm6_7_ = _mm_slli_si128(xmm6_7, 8);
            __m128i result_minpos = _mm_or_si128(xmm4_5, xmm6_7_);
            __m128i index_minpos = result_minpos;

            __m128i literal_cnt_mask = _mm_set1_epi32((int)0xFFFF0000L);
            __m128i result_minpos_only_cnt = _mm_or_si128(result_minpos, literal_cnt_mask);
            __m128i result_minpos_all = _mm_minpos_epu16(result_minpos_only_cnt);      // SSE 4.1

            uint32_t result_minpos_all32 = (uint32_t)_mm_cvtsi128_si32(result_minpos_all);
            int min_literal_cnt16 = (int)(result_minpos_all32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = result_minpos_all32 >> 17U;
                __m128i min_literal_id_sr15 = _mm_srli_epi64(result_minpos_all, 15);
                __m128i literal_index_sr16  = _mm_srli_epi32(index_minpos, 16);

                __m128i min_literal_id128 = _mm_shuffle_epi8(literal_index_sr16, min_literal_id_sr15);  // SSSE3
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(min_literal_id128);
                int min_literal_offset = (int)(min_literal_id32 & 0x000000FFUL);
                min_literal_id = index_base + min_block_index16 * 8 + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 32;
            pinfo += 64;
        }

        if ((pinfo_end - pinfo) >= 32) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pinfo + 16));

            __m128i xmm2 = _mm_minpos_epu16(xmm0);      // SSE 4.1
            __m128i xmm3 = _mm_minpos_epu16(xmm1);      // SSE 4.1

            __m128i xmm3_ = _mm_slli_epi64(xmm3, 32);
            __m128i result_minpos = _mm_blend_epi16(xmm2, xmm3_, 0b00001100);  // SSE 4.1
            __m128i index_minpos = result_minpos;

            __m128i literal_cnt_mask = _mm_set_epi32(0xFFFFFFFFL, 0xFFFFFFFFL, 0xFFFF0000L, 0xFFFF0000L);
            __m128i result_minpos_only_cnt = _mm_or_si128(result_minpos, literal_cnt_mask);
            __m128i result_minpos_all = _mm_minpos_epu16(result_minpos_only_cnt);      // SSE 4.1

            uint32_t result_minpos_all32 = (uint32_t)_mm_cvtsi128_si32(result_minpos_all);
            int min_literal_cnt16 = (int)(result_minpos_all32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_block_index16 = result_minpos_all32 >> 17U;
                __m128i min_literal_id_sr15 = _mm_srli_epi64(result_minpos_all, 15);
                __m128i literal_index_sr16 = _mm_srli_epi32(index_minpos, 16);

                __m128i min_literal_id128 = _mm_shuffle_epi8(literal_index_sr16, min_literal_id_sr15);  // SSSE3
                uint32_t min_literal_id32 = (uint32_t)_mm_cvtsi128_si32(min_literal_id128);
                int min_literal_offset = (int)(min_literal_id32 & 0x000000FFUL);
                min_literal_id = index_base + min_block_index16 * 8 + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 16;
            pinfo += 32;
        }

        if ((pinfo_end - pinfo) >= 16) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pinfo + 0));
            __m128i result_minpos = _mm_minpos_epu16(xmm0);    // SSE 4.1

            uint32_t min_literal_cnt32 = (uint32_t)_mm_cvtsi128_si32(result_minpos);
            int min_literal_cnt16 = (int)(min_literal_cnt32 & 0x0000FFFFULL);
            if (min_literal_cnt16 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt16;

                uint32_t min_literal_offset = min_literal_cnt32 >> 17U;
                min_literal_id = index_base + min_literal_offset;

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }
            index_base += 8;
            pinfo += 16;
        }

        // Last remain items (less than 8 items)
        while (pinfo < pinfo_end) {
            literal_info_t * pliteral_info = (literal_info_t *)pinfo;
            if (pliteral_info->enable == 0) {
                int literal_cnt = pliteral_info->count;
                if (literal_cnt < min_literal_cnt) {
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return index_base;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = index_base;
                }
            }
            index_base++;
            pinfo += 2;
        }

        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#elif defined(__SSE2__)

    int get_min_literal_normal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_enable_[i] == 0) {
                int literal_cnt = literal_count_[i];
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return i;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

    //
    // Horizontal minimum and maximum using SSE
    // See: https://stackoverflow.com/questions/22256525/horizontal-minimum-and-maximum-using-sse
    //
    int get_min_literal(int & out_min_literal_cnt) {
        int min_literal_cnt = 254;
        int min_literal_id = 0;
        int index_base = 0;

        const char * pcount     = (const char *)&literal_count_[0];
        const char * pcount_end = (const char *)&literal_count_[TotalLiterals];
        const char * penable    = (const char *)&literal_enable_[0];
        while ((pcount_end - pcount) >= 64) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));

            __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

            xmm0 = _mm_or_si128(xmm0, xmm2);
            xmm1 = _mm_or_si128(xmm1, xmm3);

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shufflelo_epi16(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));
            xmm1 = _mm_min_epu8(xmm1, _mm_srli_epi16(xmm1, 8));

            __m128i result_minpos_03 = _mm_min_epu8(xmm0, xmm1);

            __m128i xmm4 = _mm_load_si128((const __m128i *)(pcount + 32));
            __m128i xmm5 = _mm_load_si128((const __m128i *)(pcount + 48));

            __m128i xmm6 = _mm_load_si128((const __m128i *)(penable + 32));
            __m128i xmm7 = _mm_load_si128((const __m128i *)(penable + 48));

            xmm4 = _mm_or_si128(xmm4, xmm6);
            xmm5 = _mm_or_si128(xmm5, xmm7);

            xmm4 = _mm_min_epu8(xmm4, _mm_shuffle_epi32(xmm4, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shuffle_epi32(xmm5, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm4 = _mm_min_epu8(xmm4, _mm_shuffle_epi32(xmm4, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shuffle_epi32(xmm5, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm4 = _mm_min_epu8(xmm4, _mm_shufflelo_epi16(xmm4, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm5 = _mm_min_epu8(xmm5, _mm_shufflelo_epi16(xmm5, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm4 = _mm_min_epu8(xmm4, _mm_srli_epi16(xmm4, 8));
            xmm5 = _mm_min_epu8(xmm5, _mm_srli_epi16(xmm5, 8));

            __m128i result_minpos_47 = _mm_min_epu8(xmm4, xmm5);

            // The minimum literal count of per 64 numbers
            __m128i result_minpos_u8 = _mm_min_epu8(result_minpos_03, result_minpos_47);

            int min_literal_cnt8 = _mm_cvtsi128_si32(result_minpos_u8) & 0x000000FFL;
            if (min_literal_cnt8 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt8;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm2 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

                    xmm2 = _mm_or_si128(xmm2, xmm3);
                    xmm2 = _mm_cmpeq_epi8(xmm2, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm2);
                    if (equal_mask == 0) {
                        __m128i xmm4 = _mm_load_si128((const __m128i *)(pcount + 32));
                        __m128i xmm5 = _mm_load_si128((const __m128i *)(penable + 32));

                        xmm4 = _mm_or_si128(xmm4, xmm5);
                        xmm4 = _mm_cmpeq_epi8(xmm4, min_cmp);

                        equal_mask = _mm_movemask_epi8(xmm4);
                        if (equal_mask == 0) {
                            __m128i xmm6 = _mm_load_si128((const __m128i *)(pcount + 48));
                            __m128i xmm7 = _mm_load_si128((const __m128i *)(penable + 48));

                            xmm6 = _mm_or_si128(xmm6, xmm7);
                            xmm6 = _mm_cmpeq_epi8(xmm6, min_cmp);

                            equal_mask = _mm_movemask_epi8(xmm6);
                            if (equal_mask == 0) {
                                assert(false);
                            }
                            else {
                                int min_literal_offset = BitUtils::bsf(equal_mask);
                                min_literal_id = index_base + 3 * 16 + min_literal_offset;
                            }
                        }
                        else {
                            int min_literal_offset = BitUtils::bsf(equal_mask);
                            min_literal_id = index_base + 2 * 16 + min_literal_offset;
                        }
                    }
                    else {
                        int min_literal_offset = BitUtils::bsf(equal_mask);
                        min_literal_id = index_base + 1 * 16 + min_literal_offset;
                    }
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + 0 * 16 + min_literal_offset;
                }

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }

            index_base += 64;
            penable += 64;
            pcount += 64;
        }

        if ((pcount_end - pcount) >= 32) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(pcount + 16));

            __m128i xmm2 = _mm_load_si128((const __m128i *)(penable + 0));
            __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

            xmm0 = _mm_or_si128(xmm0, xmm2);
            xmm1 = _mm_or_si128(xmm1, xmm3);

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm1 = _mm_min_epu8(xmm1, _mm_shufflelo_epi16(xmm1, _MM_SHUFFLE(1, 1, 1, 1)));

            xmm0 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));
            xmm1 = _mm_min_epu8(xmm1, _mm_srli_epi16(xmm1, 8));

            // The minimum literal count of per 32 numbers
            __m128i result_minpos_u8 = _mm_min_epu8(xmm0, xmm1);

            int min_literal_cnt8 = _mm_cvtsi128_si32(result_minpos_u8) & 0x000000FFL;
            if (min_literal_cnt8 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt8;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    __m128i xmm2 = _mm_load_si128((const __m128i *)(pcount + 16));
                    __m128i xmm3 = _mm_load_si128((const __m128i *)(penable + 16));

                    xmm2 = _mm_or_si128(xmm2, xmm3);
                    xmm2 = _mm_cmpeq_epi8(xmm2, min_cmp);

                    equal_mask = _mm_movemask_epi8(xmm2);
                    if (equal_mask == 0) {
                        assert(false);
                    }
                    else {
                        int min_literal_offset = BitUtils::bsf(equal_mask);
                        min_literal_id = index_base + 1 * 16 + min_literal_offset;
                    }
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + 0 * 16 + min_literal_offset;
                }

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }

            index_base += 32;
            penable += 32;
            pcount += 32;
        }

        if ((pcount_end - pcount) >= 16) {
            __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
            __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

            xmm0 = _mm_or_si128(xmm0, xmm1);
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(3, 2, 3, 2)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));
            xmm0 = _mm_min_epu8(xmm0, _mm_shufflelo_epi16(xmm0, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum literal count of per 16 numbers
            __m128i result_minpos_u8 = _mm_min_epu8(xmm0, _mm_srli_epi16(xmm0, 8));

            int min_literal_cnt8 = _mm_cvtsi128_si32(result_minpos_u8) & 0x000000FFL;
            if (min_literal_cnt8 < min_literal_cnt) {
                min_literal_cnt = min_literal_cnt8;

                __m128i xmm0 = _mm_load_si128((const __m128i *)(pcount + 0));
                __m128i xmm1 = _mm_load_si128((const __m128i *)(penable + 0));

                __m128i min_cmp = _mm_set1_epi8((char)min_literal_cnt);

                xmm0 = _mm_or_si128(xmm0, xmm1);
                xmm0 = _mm_cmpeq_epi8(xmm0, min_cmp);

                int equal_mask = _mm_movemask_epi8(xmm0);
                if (equal_mask == 0) {
                    assert(false);
                }
                else {
                    int min_literal_offset = BitUtils::bsf(equal_mask);
                    min_literal_id = index_base + min_literal_offset;
                }

                if (min_literal_cnt <= kLiteralCntThreshold) {
                    out_min_literal_cnt = min_literal_cnt;
                    return min_literal_id;
                }
            }

            index_base += 16;
            penable += 16;
            pcount += 16;
        }

        // Last remain items (less than 16 items)
        while (pcount < pcount_end) {
            uint8_t * pcol_enable = (uint8_t *)penable;
            if (*pcol_enable == 0) {
                int literal_cnt = *pcount;
                if (literal_cnt < min_literal_cnt) {
                    if (literal_cnt == 0) {
                        out_min_literal_cnt = 0;
                        return index_base;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = index_base;
                }
            }
            index_base++;
            penable++;
            pcount++;
        }

        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#else

    int get_min_literal(int & out_min_literal_cnt) const {
        int min_literal_cnt = 254;
        int min_literal_id = -1;
        for (int i = 0; i < TotalLiterals; i++) {
            if (literal_enable_[i] == 0) {
                int literal_cnt = literal_count_[i];
                if (literal_cnt < min_literal_cnt) {
                    assert(literal_cnt >= 0);
                    if (literal_cnt <= kLiteralCntThreshold) {
                        out_min_literal_cnt = literal_cnt;
                        return i;
                    }
                    min_literal_cnt = literal_cnt;
                    min_literal_id = i;
                }
            }
        }
        out_min_literal_cnt = min_literal_cnt;
        return min_literal_id;
    }

#endif // __SSE4_1__

    inline void doFillNum(size_t pos, size_t row, size_t col,
                          size_t box, size_t cell, size_t num) {
        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        assert(this->num_cells_[num].test(box_pos));

        assert(this->box_cell_nums_[box][cell].test(num));
        assert(this->row_num_cols_[num][row].test(col));
        assert(this->col_num_rows_[num][col].test(row));
        assert(this->box_num_cells_[num][box].test(cell));

        disable_cell_literal(box_pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);

        size_t num_bits = this->box_cell_nums_[box][cell].to_ullong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows16 + row;
            col_idx = _num * Cols16 + col;
            box_idx = _num * Boxes16 + box;

            assert(this->num_cells_[_num].test(box_pos));

            assert(this->box_cell_nums_[box][cell].test(_num));
            assert(this->row_num_cols_[_num][row].test(col));
            assert(this->col_num_rows_[_num][col].test(row));
            assert(this->box_num_cells_[_num][box].test(cell));

            this->num_cells_[_num].reset(box_pos);

            this->box_cell_nums_[box][cell].reset(_num);
            this->row_num_cols_[_num][row].reset(col);
            this->col_num_rows_[_num][col].reset(row);
            this->box_num_cells_[_num][box].reset(cell);

            dec_cell_literal_cnt(box_pos);
            dec_row_literal_cnt(row_idx);
            dec_col_literal_cnt(col_idx);
            dec_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline void updateNeighborCellsEffect(size_t fill_pos, size_t num) {
        const PackedBitSet<BoardSize16> & neighborsMask = neighbor_boxes_mask[fill_pos];
        PackedBitSet<BoardSize16> & digitCells = this->num_cells_[num];

        PackedBitSet<BoardSize16> effect_cells = digitCells & neighborsMask;
        digitCells ^= effect_cells;

        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t box_pos = index * effect_cells.unit_bits() + bit_pos;
            const BoxesInfo & boxesInfo = SudokuTy::boxes_info16[box_pos];
            size_t box = boxesInfo.box;
            size_t cell = boxesInfo.cell;
            if (this->box_cell_nums_[box][cell].test(num)) {
                this->box_cell_nums_[box][cell].reset(num);
                dec_cell_literal_cnt(box_pos);

                size_t row = boxesInfo.row;
                size_t col = boxesInfo.col;

                size_t row_idx = num * Rows16 + row;
                size_t col_idx = num * Cols16 + col;
                size_t box_idx = num * Boxes16 + box;

                assert(this->row_num_cols_[num][row].test(col));
                assert(this->col_num_rows_[num][col].test(row));
                assert(this->box_num_cells_[num][box].test(cell));

                this->row_num_cols_[num][row].reset(col);
                this->col_num_rows_[num][col].reset(row);
                this->box_num_cells_[num][box].reset(cell);

                dec_row_literal_cnt(row_idx);
                dec_col_literal_cnt(col_idx);
                dec_box_literal_cnt(box_idx);
            }
        }
    }

    inline void doFillNum(size_t pos, size_t row, size_t col,
                          size_t box, size_t cell, size_t num,
                          PackedBitSet<Numbers16> & save_bits) {
        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        assert(this->num_cells_[num].test(box_pos));

        assert(this->box_cell_nums_[box][cell].test(num));
        assert(this->row_num_cols_[num][row].test(col));
        assert(this->col_num_rows_[num][col].test(row));
        assert(this->box_num_cells_[num][box].test(cell));

        disable_cell_literal(box_pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);

        this->num_cells_[num].reset(box_pos);

        // Save cell num bits
        save_bits = this->box_cell_nums_[box][cell];
        this->box_cell_nums_[box][cell].reset();
        this->row_num_cols_[num][row].reset(col);
        this->col_num_rows_[num][col].reset(row);
        this->box_num_cells_[num][box].reset(cell);

        size_t num_bits = save_bits.to_ulong();
        // Exclude the current number, because it has been processed.
        uint32_t n_num_bit = 1u << num;
        num_bits ^= (size_t)n_num_bit;
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows16 + row;
            col_idx = _num * Cols16 + col;
            box_idx = _num * Boxes16 + box;

            assert(this->num_cells_[_num].test(box_pos));

            assert(this->row_num_cols_[_num][row].test(col));
            assert(this->col_num_rows_[_num][col].test(row));
            assert(this->box_num_cells_[_num][box].test(cell));

            this->num_cells_[_num].reset(box_pos);

            this->row_num_cols_[_num][row].reset(col);
            this->col_num_rows_[_num][col].reset(row);
            this->box_num_cells_[_num][box].reset(cell);

            dec_row_literal_cnt(row_idx);
            dec_col_literal_cnt(col_idx);
            dec_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline void undoFillNum(size_t pos, size_t row, size_t col,
                            size_t box, size_t cell, size_t num,
                            PackedBitSet<Numbers16> & save_bits) {
        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        assert(!this->num_cells_[num].test(box_pos));

        assert(!this->box_cell_nums_[box][cell].test(num));
        assert(!this->row_num_cols_[num][row].test(col));
        assert(!this->col_num_rows_[num][col].test(row));
        assert(!this->box_num_cells_[num][box].test(cell));

        enable_cell_literal(box_pos);
        enable_row_literal(row_idx);
        enable_col_literal(col_idx);
        enable_box_literal(box_idx);

        this->num_cells_[num].set(box_pos);

        // Restore cell num bits
        this->box_cell_nums_[box][cell] = save_bits;
        this->row_num_cols_[num][row].set(col);
        this->col_num_rows_[num][col].set(row);
        this->box_num_cells_[num][box].set(cell);

        // Exclude the current number, because it has been processed.
        save_bits.reset(num);

        size_t num_bits = save_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);

            row_idx = _num * Rows16 + row;
            col_idx = _num * Cols16 + col;
            box_idx = _num * Boxes16 + box;

            assert(!this->num_cells_[_num].test(box_pos));

            assert(!this->row_num_cols_[_num][row].test(col));
            assert(!this->col_num_rows_[_num][col].test(row));
            assert(!this->box_num_cells_[_num][box].test(cell));

            this->num_cells_[_num].set(box_pos);

            this->row_num_cols_[_num][row].set(col);
            this->col_num_rows_[_num][col].set(row);
            this->box_num_cells_[_num][box].set(cell);

            inc_row_literal_cnt(row_idx);
            inc_col_literal_cnt(col_idx);
            inc_box_literal_cnt(box_idx);

            num_bits ^= num_bit;
        }
    }

    inline size_t updateNeighborCellsEffect(PackedBitSet<BoardSize16> & save_effect_cells,
                                            size_t fill_pos, size_t num) {
        const PackedBitSet<BoardSize16> & neighborsMask = neighbor_boxes_mask[fill_pos];
        PackedBitSet<BoardSize16> & digitCells = this->num_cells_[num];

        PackedBitSet<BoardSize16> effect_cells = digitCells & neighborsMask;
        digitCells ^= effect_cells;
        save_effect_cells = effect_cells;

        size_t count = 0;
        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t box_pos = index * effect_cells.unit_bits() + bit_pos;
            const BoxesInfo & boxesInfo = SudokuTy::boxes_info16[box_pos];
            size_t box = boxesInfo.box;
            size_t cell = boxesInfo.cell;
            if (this->box_cell_nums_[box][cell].test(num)) {
                this->box_cell_nums_[box][cell].reset(num);
                dec_cell_literal_cnt(box_pos);

                size_t row = boxesInfo.row;
                size_t col = boxesInfo.col;

                size_t row_idx = num * Rows16 + row;
                size_t col_idx = num * Cols16 + col;
                size_t box_idx = num * Boxes16 + box;

                assert(this->row_num_cols_[num][row].test(col));
                assert(this->col_num_rows_[num][col].test(row));
                assert(this->box_num_cells_[num][box].test(cell));

                this->row_num_cols_[num][row].reset(col);
                this->col_num_rows_[num][col].reset(row);
                this->box_num_cells_[num][box].reset(cell);

                dec_row_literal_cnt(row_idx);
                dec_col_literal_cnt(col_idx);
                dec_box_literal_cnt(box_idx);

                count++;
            }
        }

        return count;
    }

    inline size_t restoreNeighborCellsEffect(PackedBitSet<BoardSize16> & effect_cells, size_t num) {
        size_t count = 0;
        size_t cell_bit, index;
        while ((cell_bit = effect_cells.ls1b(index)) != 0) {
            size_t bit_pos = BitUtils::bsf(cell_bit);
            effect_cells.reset_bit(index, cell_bit);
            size_t box_pos = index * effect_cells.unit_bits() + bit_pos;

            const BoxesInfo & boxesInfo = SudokuTy::boxes_info16[box_pos];
            size_t box = boxesInfo.box;
            size_t cell = boxesInfo.cell;
            size_t row = boxesInfo.row;
            size_t col = boxesInfo.col;

            this->box_cell_nums_[box][cell].set(num);
            inc_cell_literal_cnt(box_pos);

            size_t row_idx = num * Rows16 + row;
            size_t col_idx = num * Cols16 + col;
            size_t box_idx = num * Boxes16 + box;

            assert(!this->row_num_cols_[num][row].test(col));
            assert(!this->col_num_rows_[num][col].test(row));
            assert(!this->box_num_cells_[num][box].test(cell));

            this->row_num_cols_[num][row].set(col);
            this->col_num_rows_[num][col].set(row);
            this->box_num_cells_[num][box].set(cell);

            inc_row_literal_cnt(row_idx);
            inc_col_literal_cnt(col_idx);
            inc_box_literal_cnt(box_idx);

            count++;
        }

        return count;
    }

    static const uint16_t kEnableLiteral16 = 0x0000;
    static const uint16_t kDisableLiteral16 = 0xFFFF;

    void _init_literal_info() {
        _init_literal_enable();
        _init_literal_count();
        _init_literal_index();
        _init_literal_total();
    }

    void _init_literal_enable() {
        for (size_t i = 0; i < Boxes16 * BoxSize16; i++) {
            this->count_.enabled.box_cells[i] = kEnableLiteral16;
        }

        for (size_t i = 0; i < Numbers16 * Rows16; i++) {
            this->count_.enabled.row_nums[i] = kEnableLiteral16;
        }

        for (size_t i = 0; i < Numbers16 * Cols16; i++) {
            this->count_.enabled.col_nums[i] = kEnableLiteral16;
        }

        for (size_t i = 0; i < Numbers16 * Boxes16; i++) {
            this->count_.enabled.box_nums[i] = kEnableLiteral16;
        }
    }

    void _init_literal_count() {
        for (size_t i = 0; i < Boxes16; i++) {
            this->count_.counts.box_cells[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.row_nums[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.col_nums[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.box_nums[i] = 255;
        }
    }

    void _init_literal_index() {
        for (size_t i = 0; i < Boxes16; i++) {
            this->count_.indexs.box_cells[i] = uint16_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.row_nums[i] = uint16_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.col_nums[i] = uint16_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.box_nums[i] = uint16_t(-1);
        }
    }

    void _init_literal_total() {
        for (size_t i = 0; i < 16; i++) {
            this->count_.total.min_literal_size[i] = 65535;
            this->count_.total.min_literal_index[i] = uint16_t(-1);
        }
    }

    // _enable_xxxx_literal()
    inline void _enable_cell_literal(size_t cell_literal) {
        this->count_.enabled.box_cells[cell_literal] = kEnableLiteral16;
    }

    inline void _enable_row_literal(size_t row_literal) {
        this->count_.enabled.row_nums[row_literal] = kEnableLiteral16;
    }

    inline void _enable_col_literal(size_t col_literal) {
        this->count_.enabled.col_nums[col_literal] = kEnableLiteral16;
    }

    inline void _enable_box_literal(size_t box_literal) {
        this->count_.enabled.box_nums[box_literal] = kEnableLiteral16;
    }

    // _disable_xxxx_literal()
    inline void _disable_cell_literal(size_t cell_literal) {
        this->count_.enabled.box_cells[cell_literal] = kDisableLiteral16;
    }

    inline void _disable_row_literal(size_t row_literal) {
        this->count_.enabled.row_nums[row_literal] = kDisableLiteral16;
    }

    inline void _disable_col_literal(size_t col_literal) {
        this->count_.enabled.col_nums[col_literal] = kDisableLiteral16;
    }

    inline void _disable_box_literal(size_t box_literal) {
        this->count_.enabled.box_nums[box_literal] = kDisableLiteral16;
    }

    inline void _doFillNum(size_t pos, size_t row, size_t col,
                           size_t box, size_t cell, size_t num) {
        assert(this->state_.box_cell_nums[box][cell].test(num));
        assert(this->state_.row_num_cols[num][row].test(col));
        assert(this->state_.col_num_rows[num][col].test(row));
        assert(this->state_.box_num_cells[num][box].test(cell));

        PackedBitSet<Numbers16> cell_num_bits = this->state_.box_cell_nums[box][cell];
        //this->state_.box_cell_nums[box][cell].fill(kAllNumbersBit);
        this->state_.box_cell_nums[box][cell].reset();

        //this->state_.row_num_cols[num][row].reset(col);
        //this->state_.col_num_rows[num][col].reset(row);
        //this->state_.box_num_cells[num][box].reset(cell);

        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        _disable_cell_literal(box_pos);
        _disable_row_literal(row_idx);
        _disable_col_literal(col_idx);
        _disable_box_literal(box_idx);

        size_t num_bits = cell_num_bits.to_ulong();
        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            assert(this->state_.row_num_cols[_num][row].test(col));
            assert(this->state_.col_num_rows[_num][col].test(row));
            assert(this->state_.box_num_cells[_num][box].test(cell));

            this->state_.row_num_cols[_num][row].reset(col);
            this->state_.col_num_rows[_num][col].reset(row);
            this->state_.box_num_cells[_num][box].reset(cell);
        }
    }

    inline void _updateNeighborCellsEffect(size_t fill_pos, size_t box, size_t num) {
        const neighbor_boxes_t & neighborBoxes = neighbor_boxes[box];
        const PackedBitSet3D<Boxes, BoxSize16, Numbers16> & neighbors_mask
            = box_cell_neighbors_mask[fill_pos][num];
        for (size_t i = 0; i < neighborBoxes.boxes_count(); i++) {
            size_t box_idx = neighborBoxes.boxes[i];
            this->state_.box_cell_nums[box_idx] &= neighbors_mask[box_idx];
        }
        //this->state_.box_cell_nums[box] &= neighbors_mask[box];

        this->state_.row_num_cols[num] &= row_neighbors_mask[fill_pos];
        this->state_.col_num_rows[num] &= col_neighbors_mask[fill_pos];
        this->state_.box_num_cells[num] &= box_num_neighbors_mask[fill_pos];
    }

    inline void _doFillNum(size_t pos, size_t row, size_t col,
                           size_t box, size_t cell, size_t num,
                           PackedBitSet<Numbers16> & save_num_bits,
                           RecoverState & recover_state) {
        assert(this->state_.box_cell_nums[box][cell].test(num));
        assert(this->state_.row_num_cols[num][row].test(col));
        assert(this->state_.col_num_rows[num][col].test(row));
        assert(this->state_.box_num_cells[num][box].test(cell));

        PackedBitSet<Numbers16> cell_num_bits = this->state_.box_cell_nums[box][cell];
        // Save cell num bits
        save_num_bits = cell_num_bits;
        //this->state_.box_cell_nums[box][cell].fill(kAllNumbersBit);
        this->state_.box_cell_nums[box][cell].reset();

        //this->state_.row_num_cols[num][row].reset(col);
        //this->state_.col_num_rows[num][col].reset(row);
        //this->state_.box_num_cells[num][box].reset(cell);

        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        _disable_cell_literal(box_pos);
        _disable_row_literal(row_idx);
        _disable_col_literal(col_idx);
        _disable_box_literal(box_idx);

        recover_state.counts.row_nums[num] = this->count_.counts.row_nums[num];
        recover_state.counts.col_nums[num] = this->count_.counts.col_nums[num];
        recover_state.counts.box_nums[num] = this->count_.counts.box_nums[num];

        size_t num_bits = cell_num_bits.to_ulong();
        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            assert(this->state_.row_num_cols[_num][row].test(col));
            assert(this->state_.col_num_rows[_num][col].test(row));
            assert(this->state_.box_num_cells[_num][box].test(cell));

            this->state_.row_num_cols[_num][row].reset(col);
            this->state_.col_num_rows[_num][col].reset(row);
            this->state_.box_num_cells[_num][box].reset(cell);

            recover_state.counts.row_nums[_num] = this->count_.counts.row_nums[_num];
            recover_state.counts.col_nums[_num] = this->count_.counts.col_nums[_num];
            recover_state.counts.box_nums[_num] = this->count_.counts.box_nums[_num];
        }
    }

    inline void _undoFillNum(size_t pos, size_t row, size_t col,
                             size_t box, size_t cell, size_t num,
                             PackedBitSet<Numbers16> & save_num_bits,
                             RecoverState & recover_state) {
        assert(!this->state_.box_cell_nums[box][cell].test(num));
        //assert(!this->state_.row_num_cols[num][row].test(col));
        //assert(!this->state_.col_num_rows[num][col].test(row));
        //assert(!this->state_.box_num_cells[num][box].test(cell));

        // Restore cell num bits
        this->state_.box_cell_nums[box][cell] = save_num_bits;

        //this->state_.row_num_cols[num][row].set(col);
        //this->state_.col_num_rows[num][col].set(row);
        //this->state_.box_num_cells[num][box].set(cell);

        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        _enable_cell_literal(box_pos);
        _enable_row_literal(row_idx);
        _enable_col_literal(col_idx);
        _enable_box_literal(box_idx);

        this->count_.counts.row_nums[num] = recover_state.counts.row_nums[num];
        this->count_.counts.col_nums[num] = recover_state.counts.col_nums[num];
        this->count_.counts.box_nums[num] = recover_state.counts.box_nums[num];

        size_t num_bits = save_num_bits.to_ulong();
        // Exclude the current number, because it has been processed.
        num_bits ^= (size_t(1) << num);
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t _num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            assert(!this->state_.row_num_cols[_num][row].test(col));
            assert(!this->state_.col_num_rows[_num][col].test(row));
            assert(!this->state_.box_num_cells[_num][box].test(cell));

            this->state_.row_num_cols[_num][row].set(col);
            this->state_.col_num_rows[_num][col].set(row);
            this->state_.box_num_cells[_num][box].set(cell);

            this->count_.counts.row_nums[_num] = recover_state.counts.row_nums[_num];
            this->count_.counts.col_nums[_num] = recover_state.counts.col_nums[_num];
            this->count_.counts.box_nums[_num] = recover_state.counts.box_nums[_num];
        }
    }

    inline void _updateNeighborCellsEffect(RecoverState & recover_state,
                                           size_t fill_pos, size_t box, size_t num) {
        // Position (Box-Cell) literal
        static const size_t boxesCount = neighbor_boxes_t::kBoxesCount;
        const neighbor_boxes_t & neighborBoxes = neighbor_boxes[box];
        const PackedBitSet3D<Boxes, BoxSize16, Numbers16> & neighbors_mask
            = box_cell_neighbors_mask[fill_pos][num];
        for (size_t i = 0; i < boxesCount; i++) {
            size_t box_idx = neighborBoxes.boxes[i];
            recover_state.boxes[i] = this->state_.box_cell_nums[box_idx];
            this->state_.box_cell_nums[box_idx] &= neighbors_mask[box_idx];

            recover_state.counts.box_cells[box_idx] = this->count_.counts.box_cells[box_idx];
        }
        //recover_state.boxes[boxesCount] = this->state_.box_cell_nums[box];
        //this->state_.box_cell_nums[box] &= neighbors_mask[box];        

        // Row literal
        recover_state.row_cols = this->state_.row_num_cols[num];
        this->state_.row_num_cols[num] &= row_neighbors_mask[fill_pos];

        // Col literal
        recover_state.col_rows = this->state_.col_num_rows[num];
        this->state_.col_num_rows[num] &= col_neighbors_mask[fill_pos];

        // Box-cell literal
        recover_state.box_cells = this->state_.box_num_cells[num];
        this->state_.box_num_cells[num] &= box_num_neighbors_mask[fill_pos];
    }

    inline void _restoreNeighborCellsEffect(const RecoverState & recover_state,
                                            size_t box, size_t num) {
        // Position (Box-Cell) literal
        static const size_t boxesCount = neighbor_boxes_t::kBoxesCount;
        const neighbor_boxes_t & neighborBoxes = neighbor_boxes[box];
        for (size_t i = 0; i < boxesCount; i++) {
            size_t box_idx = neighborBoxes.boxes[i];
            this->state_.box_cell_nums[box_idx] = recover_state.boxes[i];

            this->count_.counts.box_cells[box_idx] = recover_state.counts.box_cells[box_idx];
        }
        //this->state_.box_cell_nums[box] = recover_state.boxes[boxesCount];

        // Row literal
        this->state_.row_num_cols[num] = recover_state.row_cols;

        // Col literal
        this->state_.col_num_rows[num] = recover_state.col_rows;

        // Box-cell literal
        this->state_.box_num_cells[num] = recover_state.box_cells;
    }

    inline uint32_t count_all_literal_size(uint32_t & out_min_literal_index) {
        BitVec16x16 bitboard;

        // Position (Box-Cell) literal
        uint32_t min_cell_size = 255;
        uint32_t min_cell_index = uint32_t(-1);
        for (size_t box = 0; box < Boxes; box++) {
            const PackedBitSet2D<BoxSize16, Numbers16> * bitset;
            bitset = &this->state_.box_cell_nums[box];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Numbers>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box * BoxSize16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_cells[box * BoxSize16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Numbers>(min_cell_size, min_index);
            this->count_.counts.box_cells[box] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.box_cells[box] = (uint16_t)min_index;
            }
            else {
                size_t cell_index = box * BoxSize16 + min_index;
                this->count_.indexs.box_cells[box] = (uint16_t)cell_index;
                min_cell_index = (uint32_t)cell_index;
            }
        }
        this->count_.total.min_literal_size[0] = (uint16_t)min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            const PackedBitSet2D<Rows16, Cols16> * bitset;
            bitset = &this->state_.row_num_cols[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Cols>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.row_nums[num * Rows16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.row_nums[num * Rows16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.row_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.row_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t row_index = num * Rows16 + min_index;
                this->count_.indexs.row_nums[num] = (uint16_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
        }
        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            const PackedBitSet2D<Cols16, Rows16> * bitset;
            bitset = &this->state_.col_num_rows[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Rows>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.col_nums[num * Cols16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.col_nums[num * Cols16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.col_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.col_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t col_index = num * Cols16 + min_index;
                this->count_.indexs.col_nums[num] = (uint16_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
        }
        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            const PackedBitSet2D<Boxes16, BoxSize16> * bitset;
            bitset = &this->state_.box_num_cells[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<BoxSize>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_nums[num * Boxes16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_nums[num * Boxes16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.box_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.box_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t box_index = num * Boxes16 + min_index;
                this->count_.indexs.box_nums[num] = (uint16_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
        }
        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        uint32_t min_literal_type;

        BitVec16x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        uint32_t min_literal_size = min_literal.minpos16_and_index<4>(min_literal_type);
        uint32_t min_literal_index = min_literal_type * uint32_t(BoardSize16) +
                                     this->count_.total.min_literal_index[min_literal_type];

        out_min_literal_index = min_literal_index;
        return min_literal_size;
    }

    inline uint32_t count_delta_literal_size(uint32_t & out_min_literal_index,
                                             PackedBitSet<Numbers16> & cell_num_bits,
                                             size_t box) {
        BitVec16x16 bitboard;

        // Position (Box-Cell) literal
        uint32_t min_cell_size = 255;
        uint32_t min_cell_index = uint32_t(-1);

        // Neighbor boxes
        static const size_t boxesCount = neighbor_boxes_t::kBoxesCount;
        const neighbor_boxes_t & neighborBoxes = neighbor_boxes[box];
        for (size_t i = 0; i < boxesCount; i++) {
            size_t box_idx = neighborBoxes.boxes[i];
            const PackedBitSet2D<BoxSize16, Numbers16> * bitset;
            bitset = &this->state_.box_cell_nums[box_idx];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Numbers>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box_idx * BoxSize16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_cells[box_idx * BoxSize16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Numbers>(min_cell_size, min_index);
            this->count_.counts.box_cells[box_idx] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.box_cells[box_idx] = (uint16_t)min_index;
            }
            else {
                size_t cell_index = box_idx * BoxSize16 + min_index;
                this->count_.indexs.box_cells[box_idx] = (uint16_t)cell_index;
                min_cell_index = (uint32_t)cell_index;
            }
        }

#if 0
        // Current box
        {
            const PackedBitSet2D<BoxSize16, Numbers16> * bitset;
            bitset = &this->state_.box_cell_nums[box];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Numbers>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box * BoxSize16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_cells[box * BoxSize16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Numbers>(min_cell_size, min_index);
            this->count_.counts.box_cells[box] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.box_cells[box] = (uint16_t)min_index;
            }
            else {
                size_t cell_index = box * BoxSize16 + min_index;
                this->count_.indexs.box_cells[box] = (uint16_t)cell_index;
                min_cell_index = (uint32_t)cell_index;
            }
        }
#endif

        BitVec16x16 cell_literal_minpos;
        cell_literal_minpos.loadAligned(&this->count_.counts.box_cells[0]);
        uint32_t box_id;
        uint32_t new_min_cell_size = cell_literal_minpos.minpos16_and_index<Numbers>(box_id);
        min_cell_index = this->count_.indexs.box_cells[box_id];
        if (min_cell_index == uint32_t(uint16_t(-1))) {
            const PackedBitSet2D<BoxSize16, Numbers16> * bitset;
            bitset = &this->state_.box_cell_nums[box_id];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Numbers>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box_id * BoxSize16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_cells[box_id * BoxSize16]);
            popcnt16 |= enable_mask;

            uint32_t min_index;
            uint32_t min_size = popcnt16.minpos16_and_index<Numbers>(min_index);
            this->count_.counts.box_cells[box_id] = (uint16_t)min_size;
            assert(min_size == new_min_cell_size);
            uint32_t cell_index = box_id * uint32_t(BoxSize16) + min_index;
            this->count_.indexs.box_cells[box_id] = (uint16_t)cell_index;
            min_cell_index = cell_index;
        }

        this->count_.total.min_literal_size[0] = (uint16_t)new_min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);

        size_t num_bits = cell_num_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            const PackedBitSet2D<Rows16, Cols16> * bitset;
            bitset = &this->state_.row_num_cols[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Cols>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.row_nums[num * Rows16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.row_nums[num * Rows16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.row_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.row_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t row_index = num * Rows16 + min_index;
                this->count_.indexs.row_nums[num] = (uint16_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
        }

        BitVec16x16 row_literal_minpos;
        row_literal_minpos.loadAligned(&this->count_.counts.row_nums[0]);
        uint32_t num_index;
        min_row_size = row_literal_minpos.minpos16_and_index<Numbers>(num_index);
        min_row_index = this->count_.indexs.row_nums[num_index];
        if (min_row_index == uint32_t(uint16_t(-1))) {
            const PackedBitSet2D<Rows16, Cols16> * bitset;
            bitset = &this->state_.row_num_cols[num_index];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Cols>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.row_nums[num_index * Rows16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.row_nums[num_index * Rows16]);
            popcnt16 |= enable_mask;

            uint32_t min_index;
            uint32_t min_size = popcnt16.minpos16_and_index<Cols>(min_index);
            this->count_.counts.row_nums[num_index] = (uint16_t)min_size;
            assert(min_size == min_row_size);
            uint32_t row_index = num_index * uint32_t(Rows16) + min_index;
            this->count_.indexs.row_nums[num_index] = (uint16_t)row_index;
            min_row_index = row_index;
        }

        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);

        num_bits = cell_num_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            const PackedBitSet2D<Cols16, Rows16> * bitset;
            bitset = &this->state_.col_num_rows[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Rows>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.col_nums[num * Cols16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.col_nums[num * Cols16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.col_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.col_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t col_index = num * Cols16 + min_index;
                this->count_.indexs.col_nums[num] = (uint16_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
        }

        BitVec16x16 col_literal_minpos;
        col_literal_minpos.loadAligned(&this->count_.counts.col_nums[0]);
        min_col_size = col_literal_minpos.minpos16_and_index<Numbers>(num_index);
        min_col_index = this->count_.indexs.col_nums[num_index];
        if (min_col_index == uint32_t(uint16_t(-1))) {
            const PackedBitSet2D<Cols16, Rows16> * bitset;
            bitset = &this->state_.col_num_rows[num_index];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Rows>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.col_nums[num_index * Cols16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.col_nums[num_index * Cols16]);
            popcnt16 |= enable_mask;

            uint32_t min_index;
            uint32_t min_size = popcnt16.minpos16_and_index<Rows>(min_index);
            this->count_.counts.col_nums[num_index] = (uint16_t)min_size;
            assert(min_size == min_col_size);
            uint32_t col_index = num_index * uint32_t(Cols16) + min_index;
            this->count_.indexs.col_nums[num_index] = (uint16_t)col_index;
            min_col_index = col_index;
        }

        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        num_bits = cell_num_bits.to_ulong();
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            const PackedBitSet2D<Boxes16, BoxSize16> * bitset;
            bitset = &this->state_.box_num_cells[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<BoxSize>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_nums[num * Boxes16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_nums[num * Boxes16]);
            popcnt16 |= enable_mask;

            uint32_t min_index = uint32_t(-1);
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.box_nums[num] = (uint16_t)min_size;
            if (min_index == uint32_t(-1)) {
                this->count_.indexs.box_nums[num] = (uint16_t)min_index;
            }
            else {
                size_t box_index = num * Boxes16 + min_index;
                this->count_.indexs.box_nums[num] = (uint16_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
        }

        BitVec16x16 box_literal_minpos;
        box_literal_minpos.loadAligned(&this->count_.counts.box_nums[0]);
        min_box_size = box_literal_minpos.minpos16_and_index<Numbers>(num_index);
        min_box_index = this->count_.indexs.box_nums[num_index];
        if (min_box_index == uint32_t(uint16_t(-1))) {
            const PackedBitSet2D<Boxes16, BoxSize16> * bitset;
            bitset = &this->state_.box_num_cells[num_index];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<BoxSize>();
#if V3_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_nums[num_index * Boxes16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_nums[num_index * Boxes16]);
            popcnt16 |= enable_mask;

            uint32_t min_index;
            uint32_t min_size = popcnt16.minpos16_and_index<BoxSize>(min_index);
            this->count_.counts.box_nums[num_index] = (uint16_t)min_size;
            assert(min_size == min_box_size);
            uint32_t box_index = num_index * uint32_t(Boxes16) + min_index;
            this->count_.indexs.box_nums[num_index] = (uint16_t)box_index;
            min_box_index = box_index;
        }

        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        static const uint32_t literal_max_value[4] = { Numbers, Cols, Rows, BoxSize };

        BitVec16x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);

        uint32_t min_literal_type;
        uint32_t min_literal_size = min_literal.minpos16_and_index<4>(min_literal_type);
        uint32_t min_literal_index;
        if (min_literal_size < literal_max_value[min_literal_type]) {
            min_literal_index = min_literal_type * uint32_t(BoardSize16) +
                                this->count_.total.min_literal_index[min_literal_type];
            out_min_literal_index = min_literal_index;
        }
        else {
            min_literal_size = 0;
#ifndef NDEBUG
            min_literal_index = 0;
            out_min_literal_index = min_literal_index;
#endif
        }
        return min_literal_size;
    }

    bool verify_bitboard_state() {
#if (V3_ENABLE_OLD_ALGORITHM == 0)
        return true;
#endif
#ifdef NDEBUG
        return true;
#endif
        bool is_correct1 = (this->state_.box_cell_nums == this->box_cell_nums_);
        bool is_correct2 = (this->state_.row_num_cols == this->row_num_cols_);
        bool is_correct3 = (this->state_.col_num_rows == this->col_num_rows_);
        bool is_correct4 = (this->state_.box_num_cells == this->box_num_cells_);
        return (is_correct1 && is_correct2 && is_correct3 && is_correct4);
    }

    bool verify_literal_size() {
#if (V3_ENABLE_OLD_ALGORITHM == 0)
        return true;
#endif
        bool is_correct = true;
        return is_correct;

        // Cell (Box-Cell) literal
        for (size_t box = 0; box < Boxes; box++) {
            uint16_t * count_size = (uint16_t *)&this->count_.sizes.box_cells[box * BoxSize16];
            for (size_t cell = 0; cell < BoxSize; cell++) {
                uint8_t enable = get_cell_literal_enable(box * BoxSize16 + cell);
                if (enable != 0xFF) {
                    uint16_t size1 = get_cell_literal_cnt(box * BoxSize16 + cell);
                    uint16_t size2 = count_size[cell];
                    if (size1 != size2) {
                        return false;
                    }
                }
                else {
                    uint16_t size2 = count_size[cell];
                    if (size2 != 0) {
                        return false;
                    }
                }
            }
        }

        // Row literal
        for (size_t num = 0; num < Numbers; num++) {
            uint16_t * count_size = (uint16_t *)&this->count_.sizes.row_nums[num * Rows16];
            for (size_t row = 0; row < Rows; row++) {
                uint8_t enable = get_row_literal_enable(num, row);
                if (enable != 0xFF) {
                    uint16_t size1 = get_row_literal_cnt(num, row);
                    uint16_t size2 = count_size[row];
                    if (size1 != size2) {
                        return false;
                    }
                }
                else {
                    uint16_t size2 = count_size[row];
                    if (size2 != 0) {
                        return false;
                    }
                }
            }
        }

        // Col literal
        for (size_t num = 0; num < Numbers; num++) {
            uint16_t * count_size = (uint16_t *)&this->count_.sizes.col_nums[num * Cols16];
            for (size_t col = 0; col < Cols; col++) {
                uint8_t enable = get_col_literal_enable(num, col);
                if (enable != 0xFF) {
                    uint16_t size1 = get_col_literal_cnt(num, col);
                    uint16_t size2 = count_size[col];
                    if (size1 != size2) {
                        return false;
                    }
                }
                else {
                    uint16_t size2 = count_size[col];
                    if (size2 != 0) {
                        return false;
                    }
                }
            }
        }

        // Box-Cell literal
        for (size_t num = 0; num < Numbers; num++) {
            uint16_t * count_size = (uint16_t *)&this->count_.sizes.box_nums[num * Boxes16];
            for (size_t box = 0; box < Boxes; box++) {
                uint8_t enable = get_box_literal_enable(num, box);
                if (enable != 0xFF) {
                    uint16_t size1 = get_box_literal_cnt(num, box);
                    uint16_t size2 = count_size[box];
                    if (size1 != size2) {
                        return false;
                    }
                }
                else {
                    uint16_t size2 = count_size[box];
                    if (size2 != 0) {
                        return false;
                    }
                }
            }
        }

        return is_correct;
    }

public:
    bool solve(Board & board, size_t empties, uint32_t min_literal_size, uint32_t min_literal_index) {
        if (empties == 0) {
            if (kSearchMode > SearchMode::OneAnswer) {
                this->answers_.push_back(board);
                if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                    if (this->answers_.size() > 1)
                        return true;
                }
            }
            else {
                return true;
            }
        }

        uint32_t min_literal_id = min_literal_index;
        if (min_literal_size > 0) {
            if (min_literal_size == 1)
                num_unique_candidate++;
            else
                num_guesses++;

            PackedBitSet<Numbers16> save_bits;
            PackedBitSet<Numbers16> save_num_bits;
            PackedBitSet<BoardSize16> save_effect_cells;
            RecoverState recover_state;
            size_t pos, row, col, box, cell, num;
            uint32_t next_min_literal_size, next_min_literal_index;

            int literal_type = min_literal_id / BoardSize16;
            assert(literal_type < LiteralType::MaxLiteralType);
            switch (literal_type) {
                case LiteralType::CellNums:
                {
                    size_t box_pos = (size_t)min_literal_id - CellLiteralFirst;
                    assert(min_literal_id >= CellLiteralFirst);
                    assert(box_pos < Boxes * BoxSize16);
#if 0
                    row = pos / Cols;
                    col = pos % Cols;
                    size_t box_x = col / BoxCellsX;
                    size_t box_y = row / BoxCellsY;
                    box = box_y * BoxCountX + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell_y = row % BoxCellsY;
                    cell = cell_y * BoxCellsX + cell_x;
#else
                    const BoxesInfo & boxesInfo = SudokuTy::boxes_info16[box_pos];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    box = boxesInfo.box;
                    cell = boxesInfo.cell;
                    pos = boxesInfo.pos;
#endif
                    size_t num_bits = this->state_.box_cell_nums[box][cell].to_ulong();
#if V3_ENABLE_OLD_ALGORITHM
                    assert(this->state_.box_cell_nums[box][cell].count() == get_literal_cnt(min_literal_id));
#endif
                    while (num_bits != 0) {
                        size_t num_bit = BitUtils::ls1b(num_bits);
                        num = BitUtils::bsf(num_bit);

#if V3_ENABLE_OLD_ALGORITHM
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);
#endif
                        _doFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);
                        _updateNeighborCellsEffect(recover_state, pos, box, num);

                        bool is_correct = verify_bitboard_state();
                        assert(is_correct);

                        board.cells[pos] = (char)(num + '1');

#if V3_ENABLE_OLD_ALGORITHM
                        int next_min_literal_cnt;
                        int next_min_literal_id = get_min_literal(next_min_literal_cnt);
#endif
                        next_min_literal_size = count_delta_literal_size(next_min_literal_index, save_num_bits, box);
                        //next_min_literal_size = count_all_literal_size(next_min_literal_index);

                        bool size_is_correct = verify_literal_size();
                        assert(size_is_correct);

#if V3_ENABLE_OLD_ALGORITHM
                        assert(next_min_literal_id < TotalLiterals);
                        assert(next_min_literal_index < TotalLiterals);
                        assert(next_min_literal_size == next_min_literal_cnt || next_min_literal_cnt >= Numbers);
#endif
                        if (this->solve(board, empties - 1, next_min_literal_size, next_min_literal_index)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

#if V3_ENABLE_OLD_ALGORITHM
                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);
#endif
                        _restoreNeighborCellsEffect(recover_state, box, num);
                        _undoFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);

                        is_correct = verify_bitboard_state();
                        assert(is_correct);

                        num_bits ^= num_bit;
                    }

                    break;
                }

                case LiteralType::RowNums:
                {
                    size_t literal = (size_t)min_literal_id - RowLiteralFirst;
                    assert(min_literal_id >= RowLiteralFirst);
                    assert(literal < Numbers * Rows16);
                    num = literal / Rows16;
                    row = literal % Rows16;

                    size_t col_bits = this->state_.row_num_cols[num][row].to_ulong();
#if V3_ENABLE_OLD_ALGORITHM
                    assert(this->state_.row_num_cols[num][row].count() == get_literal_cnt(min_literal_id));
#endif
                    while (col_bits != 0) {
                        size_t col_bit = BitUtils::ls1b(col_bits);
                        col = BitUtils::bsf(col_bits);
                        pos = row * Cols + col;
#if 0
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;
#else
                        const CellInfo & cellInfo = SudokuTy::cell_info[pos];
                        box = cellInfo.box;
                        cell = cellInfo.cell;
#endif

#if V3_ENABLE_OLD_ALGORITHM
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);
#endif
                        _doFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);
                        _updateNeighborCellsEffect(recover_state, pos, box, num);

                        bool is_correct = verify_bitboard_state();
                        assert(is_correct);

                        board.cells[pos] = (char)(num + '1');

#if V3_ENABLE_OLD_ALGORITHM
                        int next_min_literal_cnt;
                        int next_min_literal_id = get_min_literal(next_min_literal_cnt);
#endif
                        next_min_literal_size = count_delta_literal_size(next_min_literal_index, save_num_bits, box);
                        //next_min_literal_size = count_all_literal_size(next_min_literal_index);

                        bool size_is_correct = verify_literal_size();
                        assert(size_is_correct);

#if V3_ENABLE_OLD_ALGORITHM
                        assert(next_min_literal_id < TotalLiterals);
                        assert(next_min_literal_index < TotalLiterals);
                        assert(next_min_literal_size == next_min_literal_cnt || next_min_literal_cnt >= Cols);
#endif
                        if (this->solve(board, empties - 1, next_min_literal_size, next_min_literal_index)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

#if V3_ENABLE_OLD_ALGORITHM
                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);
#endif
                        _restoreNeighborCellsEffect(recover_state, box, num);
                        _undoFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);

                        is_correct = verify_bitboard_state();
                        assert(is_correct);

                        col_bits ^= col_bit;
                    }

                    break;
                }

                case LiteralType::ColNums:
                {
                    size_t literal = (size_t)min_literal_id - ColLiteralFirst;
                    assert(min_literal_id >= ColLiteralFirst);
                    assert(literal < Numbers * Cols16);
                    num = literal / Cols16;
                    col = literal % Cols16;

                    size_t row_bits = this->state_.col_num_rows[num][col].to_ulong();
#if V3_ENABLE_OLD_ALGORITHM
                    assert(this->state_.col_num_rows[num][col].count() == get_literal_cnt(min_literal_id));
#endif
                    while (row_bits != 0) {
                        size_t row_bit = BitUtils::ls1b(row_bits);
                        row = BitUtils::bsf(row_bits);
                        pos = row * Cols + col;
#if 0
                        size_t box_x = col / BoxCellsX;
                        size_t box_y = row / BoxCellsY;
                        box = box_y * BoxCountX + box_x;
                        size_t cell_x = col % BoxCellsX;
                        size_t cell_y = row % BoxCellsY;
                        cell = cell_y * BoxCellsX + cell_x;
#else
                        const CellInfo & cellInfo = SudokuTy::cell_info[pos];
                        box = cellInfo.box;
                        cell = cellInfo.cell;
#endif

#if V3_ENABLE_OLD_ALGORITHM
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);
#endif
                        _doFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);
                        _updateNeighborCellsEffect(recover_state, pos, box, num);

                        bool is_correct = verify_bitboard_state();
                        assert(is_correct);

                        board.cells[pos] = (char)(num + '1');

#if V3_ENABLE_OLD_ALGORITHM
                        int next_min_literal_cnt;
                        int next_min_literal_id = get_min_literal(next_min_literal_cnt);
#endif
                        next_min_literal_size = count_delta_literal_size(next_min_literal_index, save_num_bits, box);
                        //next_min_literal_size = count_all_literal_size(next_min_literal_index);

                        bool size_is_correct = verify_literal_size();
                        assert(size_is_correct);

#if V3_ENABLE_OLD_ALGORITHM
                        assert(next_min_literal_id < TotalLiterals);
                        assert(next_min_literal_index < TotalLiterals);
                        assert(next_min_literal_size == next_min_literal_cnt || next_min_literal_cnt >= Rows);
#endif
                        if (this->solve(board, empties - 1, next_min_literal_size, next_min_literal_index)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

#if V3_ENABLE_OLD_ALGORITHM
                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);
#endif
                        _restoreNeighborCellsEffect(recover_state, box, num);
                        _undoFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);

                        is_correct = verify_bitboard_state();
                        assert(is_correct);

                        row_bits ^= row_bit;
                    }

                    break;
                }

               case LiteralType::BoxNums:
                {
                    size_t literal = (size_t)min_literal_id - BoxLiteralFirst;
                    assert(min_literal_id >= BoxLiteralFirst);
                    assert(literal < Numbers * Boxes16);
                    num = literal / Boxes16;
                    box = literal % Boxes16;

                    size_t cell_bits = this->state_.box_num_cells[num][box].to_ulong();
#if V3_ENABLE_OLD_ALGORITHM
                    assert(this->state_.box_num_cells[num][box].count() == get_literal_cnt(min_literal_id));
#endif
                    while (cell_bits != 0) {
                        size_t cell_bit = BitUtils::ls1b(cell_bits);
                        cell = BitUtils::bsf(cell_bits);
#if 0
                        row = (box / BoxCountX) * BoxCellsY + (cell / BoxCellsX);
                        col = (box % BoxCountX) * BoxCellsX + (cell % BoxCellsX);
                        pos = row * Cols + col;
#else
                        const BoxesInfo & boxesInfo = SudokuTy::boxes_info16[box * BoxSize16 + cell];
                        row = boxesInfo.row;
                        col = boxesInfo.col;
                        pos = boxesInfo.pos;
#endif

#if V3_ENABLE_OLD_ALGORITHM
                        doFillNum(pos, row, col, box, cell, num, save_bits);
                        size_t effect_count = updateNeighborCellsEffect(save_effect_cells, pos, num);
#endif
                        _doFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);
                        _updateNeighborCellsEffect(recover_state, pos, box, num);

                        bool is_correct = verify_bitboard_state();
                        assert(is_correct);

                        board.cells[pos] = (char)(num + '1');

#if V3_ENABLE_OLD_ALGORITHM
                        int next_min_literal_cnt;
                        int next_min_literal_id = get_min_literal(next_min_literal_cnt);
#endif
                        next_min_literal_size = count_delta_literal_size(next_min_literal_index, save_num_bits, box);
                        //next_min_literal_size = count_all_literal_size(next_min_literal_index);

                        bool size_is_correct = verify_literal_size();
                        assert(size_is_correct);

#if V3_ENABLE_OLD_ALGORITHM
                        assert(next_min_literal_id < TotalLiterals);
                        assert(next_min_literal_index < TotalLiterals);
                        assert(next_min_literal_size == next_min_literal_cnt || next_min_literal_cnt >= BoxSize);
#endif
                        if (this->solve(board, empties - 1, next_min_literal_size, next_min_literal_index)) {
                            if (kSearchMode == SearchMode::OneAnswer) {
                                return true;
                            }
                            else if (kSearchMode == SearchMode::MoreThanOneAnswer) {
                                if (this->answers_.size() > 1)
                                    return true;
                            }
                        }

#if V3_ENABLE_OLD_ALGORITHM
                        undoFillNum(pos, row, col, box, cell, num, save_bits);
                        this->num_cells_[num] |= save_effect_cells;
                        size_t r_effect_count = restoreNeighborCellsEffect(save_effect_cells, num);
                        assert(effect_count == r_effect_count);
#endif
                        _restoreNeighborCellsEffect(recover_state, box, num);
                        _undoFillNum(pos, row, col, box, cell, num, save_num_bits, recover_state);

                        is_correct = verify_bitboard_state();
                        assert(is_correct);

                        cell_bits ^= cell_bit;
                    }

                    break;
                }

                default:
                    assert(false);
                    break;
            }
        }
        else {
            num_failed_return++;
        }

        return false;
    }

    bool solve(Board & board) {
        this->init_board(board);
        bool success = this->solve(board, this->empties_,
                                   this->count_.min_literal_size,
                                   this->count_.min_literal_index);
        return success;
    }

    void display_board(Board & board) {
        SudokuTy::display_board(board, true);
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        if (print_answer) {
            if (kSearchMode > SearchMode::OneAnswer)
                SudokuTy::display_boards(this->answers_);
            else
                SudokuTy::display_board(board);
        }
        printf("elapsed time: %0.3f ms, recur_counter: %" PRIuPTR "\n\n"
                "num_guesses: %" PRIuPTR ", num_failed_return: %" PRIuPTR ", num_unique_candidate: %" PRIuPTR "\n"
                "guess %% = %0.1f %%, failed_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
                elapsed_time, solver_type::get_total_search_counter(),
                solver_type::get_num_guesses(),
                solver_type::get_num_failed_return(),
                solver_type::get_num_unique_candidate(),
                solver_type::get_guess_percent(),
                solver_type::get_failed_return_percent(),
                solver_type::get_unique_candidate_percent());
    }
};

template <typename SudokuTy>
bool Solver<SudokuTy>::is_mask_inited = false;

template <typename SudokuTy>
std::vector<typename Solver<SudokuTy>::neighbor_boxes_t>
Solver<SudokuTy>::neighbor_boxes;

template <typename SudokuTy>
alignas(32)
PackedBitSet2D<Solver<SudokuTy>::BoardSize, Solver<SudokuTy>::Rows16 * Solver<SudokuTy>::Cols16>
Solver<SudokuTy>::neighbor_cells_mask;

template <typename SudokuTy>
alignas(32)
PackedBitSet2D<Solver<SudokuTy>::BoardSize, Solver<SudokuTy>::Boxes16 * Solver<SudokuTy>::BoxSize16>
Solver<SudokuTy>::neighbor_boxes_mask;

template <typename SudokuTy>
alignas(32)
PackedBitSet3D<Solver<SudokuTy>::Boxes, Solver<SudokuTy>::BoxSize16, Solver<SudokuTy>::Numbers16>
Solver<SudokuTy>::box_cell_neighbors_mask[Solver<SudokuTy>::BoardSize][Solver<SudokuTy>::Numbers];

template <typename SudokuTy>
alignas(32)
PackedBitSet3D<Solver<SudokuTy>::BoardSize, Solver<SudokuTy>::Rows16, Solver<SudokuTy>::Cols16>
Solver<SudokuTy>::row_neighbors_mask;

template <typename SudokuTy>
alignas(32)
PackedBitSet3D<Solver<SudokuTy>::BoardSize, Solver<SudokuTy>::Cols16, Solver<SudokuTy>::Rows16>
Solver<SudokuTy>::col_neighbors_mask;

template <typename SudokuTy>
alignas(32)
PackedBitSet3D<Solver<SudokuTy>::BoardSize, Solver<SudokuTy>::Boxes16, Solver<SudokuTy>::BoxSize16>
Solver<SudokuTy>::box_num_neighbors_mask;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_guesses = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_unique_candidate = 0;

template <typename SudokuTy>
size_t Solver<SudokuTy>::num_failed_return = 0;

} // namespace v3
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V3_H
