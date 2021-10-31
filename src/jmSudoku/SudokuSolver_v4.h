
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

    static const size_t kAllRowsBits = SudokuTy::kAllRowsBits;
    static const size_t kAllColsBits = SudokuTy::kAllColsBits;
    static const size_t kAllBoxesBits = SudokuTy::kAllBoxesBits;
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

    struct Count {
        alignas(16) uint8_t num_boxes[Numbers][Boxes16];
        alignas(16) uint8_t num_rows[Numbers][Rows16];
        alignas(16) uint8_t num_cols[Numbers][Cols16];
    };

    struct LiteralInfo {
        uint32_t literal_index;
        uint32_t literal_size;
    };

    union literal_count_t {
        struct {
            uint8_t count:7;
            uint8_t enable:1;
        };
        uint8_t value;
    };

#pragma pack(pop)

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct NeighborBoxes {
        static const uint32_t kBoxesCount = (uint32_t)((nBoxCountX - 1) + (nBoxCountY - 1) + 1);
        uint32_t boxes_count() const { return kBoxesCount; }

        int boxes[kBoxesCount];
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct HVNeighborBoxes {
        static const uint32_t kHorizontalBoxes = uint32_t(nBoxCountX - 1);
        static const uint32_t kVerticalBoxes = uint32_t(nBoxCountX - 1);
        uint32_t h_boxes_count() const { return kHorizontalBoxes; }
        uint32_t v_boxes_count() const { return kVerticalBoxes; }

        int h_boxes[kHorizontalBoxes];
        int v_boxes[kVerticalBoxes];
    };

    typedef NeighborBoxes<BoxCountX, BoxCountY>     neighbor_boxes_t;
    typedef HVNeighborBoxes<BoxCountX, BoxCountY>   hv_neighbor_boxes_t;

    State           state_;
    Count           count_;
    LiteralInfo     min_info_;

    alignas(32) literal_count_t literal_info_[TotalLiterals];

    std::vector<Board>  answers_;
    size_t  empties_;

    static bool mask_is_inited;
    static std::vector<neighbor_boxes_t>    neighbor_boxes;
    static std::vector<hv_neighbor_boxes_t> hv_neighbor_boxes;

    static PackedBitSet2D<BoardSize, Rows16 * Cols16>     neighbor_cells_mask;
    static PackedBitSet2D<BoardSize, Boxes16 * BoxSize16> neighbor_boxes_mask;

    static PackedBitSet3D<Boxes, BoxSize16, Numbers16>    box_cell_neighbors_mask[BoardSize][Numbers];
    static PackedBitSet3D<BoardSize, Rows16, Cols16>      row_neighbors_mask;
    static PackedBitSet3D<BoardSize, Cols16, Rows16>      col_neighbors_mask;
    static PackedBitSet3D<BoardSize, Boxes16, BoxSize16>  box_num_neighbors_mask;

public:
    Solver() : empties_(0) {
        if (!mask_is_inited) {
            init_mask();
            mask_is_inited = true;
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
                uint32_t box = uint32_t(box_y_base + box_x);
                size_t index = 0;
                neighbor_boxes_t neighborBoxes;
                neighborBoxes.boxes[index++] = box;
                for (size_t box_i = 0; box_i < BoxCellsX; box_i++) {
                    if (box_i != box_x) {
                        neighborBoxes.boxes[index++] = uint32_t(box_y * BoxCellsX + box_i);
                    }
                }
                for (size_t box_j = 0; box_j < BoxCellsY; box_j++) {
                    if (box_j != box_y) {
                        neighborBoxes.boxes[index++] = uint32_t(box_j * BoxCellsX + box_x);
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
        this->state_.box_cell_nums.fill(kAllNumbersBits);
        this->state_.num_row_cols.fill(kAllColsBits);
        this->state_.num_col_rows.fill(kAllRowsBits);
        this->state_.num_box_cells.fill(kAllBoxSizeBits);

        num_guesses = 0;
        num_unique_candidate = 0;
        num_failed_return = 0;
        if (kSearchMode > SEARCH_MODE_ONE_ANSWER) {
            this->answers_.clear();
        }

        size_t empties = this->calc_empties(board);
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

                    this->fillNum(this->state_, pos, row, col, box, cell, num);
                    this->updateNeighborCellsEffect(this->state_, pos, box, num);
                }
                pos++;
            }
        }
        assert(pos == BoardSize);

        uint32_t min_literal_index;
        uint32_t min_literal_size = this->count_all_literal_size(min_literal_index);
        this->min_info_.literal_size = min_literal_size;
        this->min_info_.literal_index = min_literal_index;
    }

    inline void fillNum(State & state, size_t pos, size_t row, size_t col,
                        size_t box, size_t cell, size_t num) {
        assert(state.box_cell_nums[box][cell].test(num));
        assert(state.num_row_cols[num][row].test(col));
        assert(state.num_col_rows[num][col].test(row));
        assert(state.num_box_cells[num][box].test(cell));

        PackedBitSet<Numbers16> cell_num_bits = state.box_cell_nums[box][cell];
        //state.box_cell_nums[box][cell].fill(kAllNumbersBits);
        state.box_cell_nums[box][cell].reset();

        //state.row_num_cols[num][row].reset(col);
        //state.col_num_rows[num][col].reset(row);
        //state.box_num_cells[num][box].reset(cell);

        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        /*
        disable_cell_literal(box_pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);
        //*/

        size_t num_bits = cell_num_bits.to_ulong();
        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t cur_num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            assert(this->state_.num_row_cols[cur_num][row].test(col));
            assert(this->state_.num_col_rows[cur_num][col].test(row));
            assert(this->state_.num_box_cells[cur_num][box].test(cell));

            this->state_.num_row_cols[cur_num][row].reset(col);
            this->state_.num_col_rows[cur_num][col].reset(row);
            this->state_.num_box_cells[cur_num][box].reset(cell);
        }
    }

    inline void updateNeighborCellsEffect(State & state, size_t fill_pos, size_t box, size_t num) {
        const neighbor_boxes_t & neighborBoxes = neighbor_boxes[box];
        const PackedBitSet3D<Boxes, BoxSize16, Numbers16> & neighbors_mask
            = box_cell_neighbors_mask[fill_pos][num];
        for (size_t i = 0; i < neighborBoxes.boxes_count(); i++) {
            size_t box_idx = neighborBoxes.boxes[i];
            state.box_cell_nums[box_idx] &= neighbors_mask[box_idx];
        }
        //state.box_cell_nums[box] &= neighbors_mask[box];

        state.num_row_cols[num] &= row_neighbors_mask[fill_pos];
        state.num_col_rows[num] &= col_neighbors_mask[fill_pos];
        state.num_box_cells[num] &= box_num_neighbors_mask[fill_pos];
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
#if V3E_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box * BoxSize16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_cells[box * BoxSize16]);
            popcnt16 |= enable_mask;

            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Numbers>(min_cell_size, min_index);
            this->count_.counts.box_cells[box] = (uint16_t)min_size;
            if (min_index == -1) {
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
            bitset = &this->state_.num_row_cols[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Cols>();
#if V3E_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.row_nums[num * Rows16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.row_nums[num * Rows16]);
            popcnt16 |= enable_mask;

            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.row_nums[num] = (uint16_t)min_size;
            if (min_index == -1) {
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
            bitset = &this->state_.num_col_rows[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<Rows>();
#if V3E_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.col_nums[num * Cols16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.col_nums[num * Cols16]);
            popcnt16 |= enable_mask;

            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.col_nums[num] = (uint16_t)min_size;
            if (min_index == -1) {
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
            bitset = &this->state_.num_box_cells[num];
            bitboard.loadAligned(bitset);

            BitVec16x16 popcnt16 = bitboard.popcount16<BoxSize>();
#if V3E_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_nums[num * Boxes16]);
#endif
            BitVec16x16 enable_mask;
            enable_mask.loadAligned(&this->count_.enabled.box_nums[num * Boxes16]);
            popcnt16 |= enable_mask;

            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.box_nums[num] = (uint16_t)min_size;
            if (min_index == -1) {
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

        int min_literal_type;

        BitVec16x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        uint32_t min_literal_size = min_literal.minpos16_and_index<4>(min_literal_type);
        uint32_t min_literal_index = min_literal_type * uint32_t(BoardSize16) +
                                     this->count_.total.min_literal_index[min_literal_type];

        out_min_literal_index = min_literal_index;
        return min_literal_size;
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

        return false;
    }

    bool solve(Board & board) {
        this->init_board(board);
        bool success = this->solve(board, this->empties_,
                                   this->min_info_.literal_size,
                                   this->min_info_.literal_index);
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
bool Solver<SudokuTy>::mask_is_inited = false;

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

} // namespace v4
} // namespace jmSudoku

#endif // JM_SUDOKU_SOLVER_V4_H
