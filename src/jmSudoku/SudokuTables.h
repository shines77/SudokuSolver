
#ifndef JM_SUDOKU_TABLES_H
#define JM_SUDOKU_TABLES_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <bitset>
#include <cstring>      // For std::memset()
#include <type_traits>
#include <algorithm>    // For std::sort()

namespace jmSudoku {

/*
    +----------+----------+----------+
    | A1 A2 A3 | A4 A5 A6 | A7 A8 A9 |
    | B1 B2 B3 | B4 B5 B6 | B7 B8 B9 |
    | C1 C2 C3 | C4 C5 C6 | C7 C8 C9 |
    +----------+----------+----------+
    | D1 D2 D3 | D4 D5 D6 | D7 D8 D9 |
    | E1 E2 E3 | E4 E5 E6 | E7 E8 E9 |
    | F1 F2 F3 | F4 F5 F6 | F7 F8 F9 |
    +----------+----------+----------+
    | G1 G2 G3 | G4 G5 G6 | G7 G8 G9 |
    | H1 H2 H3 | H4 H5 H6 | H7 H8 H9 |
    | I1 I2 I3 | I4 I5 I6 | I7 I8 I9 |
    +----------+----------+----------+
*/

static const size_t _0 = 0;
static const size_t _1U = 1U;

static const size_t _0_ = 0;

/*
    +----------+----------+----------+
    | A1 A2 A3 | A4 A5 A6 | A7 A8 A9 |
    | B1 B2 B3 | B4 B5 B6 | B7 B8 B9 |
    | C1 C2 C3 | C4 C5 C6 | C7 C8 C9 |
    +----------+----------+----------+
*/
static const size_t A1_first = 0;
static const size_t A1 = _1U << (A1_first + 0);
static const size_t A2 = _1U << (A1_first + 1);
static const size_t A3 = _1U << (A1_first + 2);
static const size_t A4 = _1U << (A1_first + 3);
static const size_t A5 = _1U << (A1_first + 4);
static const size_t A6 = _1U << (A1_first + 5);
static const size_t A7 = _1U << (A1_first + 6);
static const size_t A8 = _1U << (A1_first + 7);
static const size_t A9 = _1U << (A1_first + 8);

static const size_t B1_first = 9;
static const size_t B1 = _1U << (B1_first + 0);
static const size_t B2 = _1U << (B1_first + 1);
static const size_t B3 = _1U << (B1_first + 2);
static const size_t B4 = _1U << (B1_first + 3);
static const size_t B5 = _1U << (B1_first + 4);
static const size_t B6 = _1U << (B1_first + 5);
static const size_t B7 = _1U << (B1_first + 6);
static const size_t B8 = _1U << (B1_first + 7);
static const size_t B9 = _1U << (B1_first + 8);

static const size_t C1_first = 18;
static const size_t C1 = _1U << (C1_first + 0);
static const size_t C2 = _1U << (C1_first + 1);
static const size_t C3 = _1U << (C1_first + 2);
static const size_t C4 = _1U << (C1_first + 3);
static const size_t C5 = _1U << (C1_first + 4);
static const size_t C6 = _1U << (C1_first + 5);
static const size_t C7 = _1U << (C1_first + 6);
static const size_t C8 = _1U << (C1_first + 7);
static const size_t C9 = _1U << (C1_first + 8);

/*
    +----------+----------+----------+
    | D1 D2 D3 | D4 D5 D6 | D7 D8 D9 |
    | E1 E2 E3 | E4 E5 E6 | E7 E8 E9 |
    | F1 F2 F3 | F4 F5 F6 | F7 F8 F9 |
    +----------+----------+----------+
*/

static const size_t D1_first = 27;
static const size_t D1 = _1U << (D1_first + 0);
static const size_t D2 = _1U << (D1_first + 1);
static const size_t D3 = _1U << (D1_first + 2);
static const size_t D4 = _1U << (D1_first + 3);
static const size_t D5 = _1U << (D1_first + 4);
static const size_t D6 = _1U << (D1_first + 5);
static const size_t D7 = _1U << (D1_first + 6);
static const size_t D8 = _1U << (D1_first + 7);
static const size_t D9 = _1U << (D1_first + 8);

static const size_t E1_first = 36;
static const size_t E1 = _1U << (E1_first + 0);
static const size_t E2 = _1U << (E1_first + 1);
static const size_t E3 = _1U << (E1_first + 2);
static const size_t E4 = _1U << (E1_first + 3);
static const size_t E5 = _1U << (E1_first + 4);
static const size_t E6 = _1U << (E1_first + 5);
static const size_t E7 = _1U << (E1_first + 6);
static const size_t E8 = _1U << (E1_first + 7);
static const size_t E9 = _1U << (E1_first + 8);

static const size_t F1_first = 45;
static const size_t F1 = _1U << (F1_first + 0);
static const size_t F2 = _1U << (F1_first + 1);
static const size_t F3 = _1U << (F1_first + 2);
static const size_t F4 = _1U << (F1_first + 3);
static const size_t F5 = _1U << (F1_first + 4);
static const size_t F6 = _1U << (F1_first + 5);
static const size_t F7 = _1U << (F1_first + 6);
static const size_t F8 = _1U << (F1_first + 7);
static const size_t F9 = _1U << (F1_first + 8);

/*
    +----------+----------+----------+
    | G1 G2 G3 | G4 G5 G6 | G7 G8 G9 |
    | H1 H2 H3 | H4 H5 H6 | H7 H8 H9 |
    | I1 I2 I3 | I4 I5 I6 | I7 I8 I9 |
    +----------+----------+----------+
*/

static const size_t G1_first = 54;
static const size_t G1 = _1U << (G1_first + 0);
static const size_t G2 = _1U << (G1_first + 1);
static const size_t G3 = _1U << (G1_first + 2);
static const size_t G4 = _1U << (G1_first + 3);
static const size_t G5 = _1U << (G1_first + 4);
static const size_t G6 = _1U << (G1_first + 5);
static const size_t G7 = _1U << (G1_first + 6);
static const size_t G8 = _1U << (G1_first + 7);
static const size_t G9 = _1U << (G1_first + 8);

static const size_t H1_first_low = 63;
static const size_t H1_first = 0;
static const size_t H1 = _1U << (H1_first_low + 0);
static const size_t H2 = _1U << (H1_first + 1);
static const size_t H3 = _1U << (H1_first + 2);
static const size_t H4 = _1U << (H1_first + 3);
static const size_t H5 = _1U << (H1_first + 4);
static const size_t H6 = _1U << (H1_first + 5);
static const size_t H7 = _1U << (H1_first + 6);
static const size_t H8 = _1U << (H1_first + 7);
static const size_t H9 = _1U << (H1_first + 8);

static const size_t I1_first = 9;
static const size_t I1 = _1U << (I1_first + 0);
static const size_t I2 = _1U << (I1_first + 1);
static const size_t I3 = _1U << (I1_first + 2);
static const size_t I4 = _1U << (I1_first + 3);
static const size_t I5 = _1U << (I1_first + 4);
static const size_t I6 = _1U << (I1_first + 5);
static const size_t I7 = _1U << (I1_first + 6);
static const size_t I8 = _1U << (I1_first + 7);
static const size_t I9 = _1U << (I1_first + 8);

struct bitmask_9x9 {
    uint64_t low;
    uint64_t high;
};

#define BitMask_9x9( \
        _A1_, _A2_, _A3_, _A4_, _A5_, _A6_, _A7_, _A8_, _A9_, \
        _B1_, _B2_, _B3_, _B4_, _B5_, _B6_, _B7_, _B8_, _B9_, \
        _C1_, _C2_, _C3_, _C4_, _C5_, _C6_, _C7_, _C8_, _C9_, \
        _D1_, _D2_, _D3_, _D4_, _D5_, _D6_, _D7_, _D8_, _D9_, \
        _E1_, _E2_, _E3_, _E4_, _E5_, _E6_, _E7_, _E8_, _E9_, \
        _F1_, _F2_, _F3_, _F4_, _F5_, _F6_, _F7_, _F8_, _F9_, \
        _G1_, _G2_, _G3_, _G4_, _G5_, _G6_, _G7_, _G8_, _G9_, \
        _H1_, _H2_, _H3_, _H4_, _H5_, _H6_, _H7_, _H8_, _H9_, \
        _I1_, _I2_, _I3_, _I4_, _I5_, _I6_, _I7_, _I8_, _I9_) \
    { \
        (_A1_ | _A2_ | _A3_ | _A4_ | _A5_ | _A6_ | _A7_ | _A8_ | _A9_) | \
        (_B1_ | _B2_ | _B3_ | _B4_ | _B5_ | _B6_ | _B7_ | _B8_ | _B9_) | \
        (_C1_ | _C2_ | _C3_ | _C4_ | _C5_ | _C6_ | _C7_ | _C8_ | _C9_) | \
        (_D1_ | _D2_ | _D3_ | _D4_ | _D5_ | _D6_ | _D7_ | _D8_ | _D9_) | \
        (_E1_ | _E2_ | _E3_ | _E4_ | _E5_ | _E6_ | _E7_ | _E8_ | _E9_) | \
        (_F1_ | _F2_ | _F3_ | _F4_ | _F5_ | _F6_ | _F7_ | _F8_ | _F9_) | \
        (_G1_ | _G2_ | _G3_ | _G4_ | _G5_ | _G6_ | _G7_ | _G8_ | _G9_) | \
        _H1_, \
        (_0_  | _H2_ | _H3_ | _H4_ | _H5_ | _H6_ | _H7_ | _H8_ | _H9_) | \
        (_I1_ | _I2_ | _I3_ | _I4_ | _I5_ | _I6_ | _I7_ | _I8_ | _I9_)   \
    }

struct BitTables {
    const bitmask_9x9 effect_mask[9] = {
        /*
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    B1, B2, B3, B4, B5, B6, B7, B8, B9,
                    C1, C2, C3, C4, C5, C6, C7, C8, C9,
                    D1, D2, D3, D4, D5, D6, D7, D8, D9,
                    E1, E2, E3, E4, E5, E6, E7, E8, E9,
                    F1, F2, F3, F4, F5, F6, F7, F8, F9,
                    G1, G2, G3, G4, G5, G6, G7, G8, G9,
                    H1, H2, H3, H4, H5, H6, H7, H8, H9,
                    I1, I2, I3, I4, I5, I6, I7, I8, I9),
        */
        // A1
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    B1, B2, B3, _0, _0, _0, _0, _0, _0,
                    C1, C2, C3, _0, _0, _0, _0, _0, _0,
                    D1, _0, _0, _0, _0, _0, _0, _0, _0,
                    E1, _0, _0, _0, _0, _0, _0, _0, _0,
                    F1, _0, _0, _0, _0, _0, _0, _0, _0,
                    G1, _0, _0, _0, _0, _0, _0, _0, _0,
                    H1, _0, _0, _0, _0, _0, _0, _0, _0,
                    I1, _0, _0, _0, _0, _0, _0, _0, _0),

        // A2
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    B1, B2, B3, _0, _0, _0, _0, _0, _0,
                    C1, C2, C3, _0, _0, _0, _0, _0, _0,
                    _0, D2, _0, _0, _0, _0, _0, _0, _0,
                    _0, E2, _0, _0, _0, _0, _0, _0, _0,
                    _0, F2, _0, _0, _0, _0, _0, _0, _0,
                    _0, G2, _0, _0, _0, _0, _0, _0, _0,
                    _0, H2, _0, _0, _0, _0, _0, _0, _0,
                    _0, I2, _0, _0, _0, _0, _0, _0, _0),

        // A3
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    B1, B2, B3, _0, _0, _0, _0, _0, _0,
                    C1, C2, C3, _0, _0, _0, _0, _0, _0,
                    _0, _0, D3, _0, _0, _0, _0, _0, _0,
                    _0, _0, E3, _0, _0, _0, _0, _0, _0,
                    _0, _0, F3, _0, _0, _0, _0, _0, _0,
                    _0, _0, G3, _0, _0, _0, _0, _0, _0,
                    _0, _0, H3, _0, _0, _0, _0, _0, _0,
                    _0, _0, I3, _0, _0, _0, _0, _0, _0),

        // A4
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, B4, B5, B6, _0, _0, _0,
                    _0, _0, _0, C4, C5, C6, _0, _0, _0,
                    _0, _0, _0, D4, _0, _0, _0, _0, _0,
                    _0, _0, _0, E4, _0, _0, _0, _0, _0,
                    _0, _0, _0, F4, _0, _0, _0, _0, _0,
                    _0, _0, _0, G4, _0, _0, _0, _0, _0,
                    _0, _0, _0, H4, _0, _0, _0, _0, _0,
                    _0, _0, _0, I4, _0, _0, _0, _0, _0),

        // A5
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, B4, B5, B6, _0, _0, _0,
                    _0, _0, _0, C4, C5, C6, _0, _0, _0,
                    _0, _0, _0, _0, D5, _0, _0, _0, _0,
                    _0, _0, _0, _0, E5, _0, _0, _0, _0,
                    _0, _0, _0, _0, F5, _0, _0, _0, _0,
                    _0, _0, _0, _0, G5, _0, _0, _0, _0,
                    _0, _0, _0, _0, H5, _0, _0, _0, _0,
                    _0, _0, _0, _0, I5, _0, _0, _0, _0),

        // A6
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, B4, B5, B6, _0, _0, _0,
                    _0, _0, _0, C4, C5, C6, _0, _0, _0,
                    _0, _0, _0, _0, _0, D6, _0, _0, _0,
                    _0, _0, _0, _0, _0, E6, _0, _0, _0,
                    _0, _0, _0, _0, _0, F6, _0, _0, _0,
                    _0, _0, _0, _0, _0, G6, _0, _0, _0,
                    _0, _0, _0, _0, _0, H6, _0, _0, _0,
                    _0, _0, _0, _0, _0, I6, _0, _0, _0),

        // A7
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, _0, _0, _0, B7, B8, B9,
                    _0, _0, _0, _0, _0, _0, C7, C8, C9,
                    _0, _0, _0, _0, _0, _0, D7, _0, _0,
                    _0, _0, _0, _0, _0, _0, E7, _0, _0,
                    _0, _0, _0, _0, _0, _0, F7, _0, _0,
                    _0, _0, _0, _0, _0, _0, G7, _0, _0,
                    _0, _0, _0, _0, _0, _0, H7, _0, _0,
                    _0, _0, _0, _0, _0, _0, I7, _0, _0),

        // A8
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, _0, _0, _0, B7, B8, B9,
                    _0, _0, _0, _0, _0, _0, C7, C8, C9,
                    _0, _0, _0, _0, _0, _0, _0, D8, _0,
                    _0, _0, _0, _0, _0, _0, _0, E8, _0,
                    _0, _0, _0, _0, _0, _0, _0, F8, _0,
                    _0, _0, _0, _0, _0, _0, _0, G8, _0,
                    _0, _0, _0, _0, _0, _0, _0, H8, _0,
                    _0, _0, _0, _0, _0, _0, _0, I8, _0),

        // A9
        BitMask_9x9(A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    _0, _0, _0, _0, _0, _0, B7, B8, B9,
                    _0, _0, _0, _0, _0, _0, C7, C8, C9,
                    _0, _0, _0, _0, _0, _0, _0, _0, D9,
                    _0, _0, _0, _0, _0, _0, _0, _0, E9,
                    _0, _0, _0, _0, _0, _0, _0, _0, F9,
                    _0, _0, _0, _0, _0, _0, _0, _0, G9,
                    _0, _0, _0, _0, _0, _0, _0, _0, H9,
                    _0, _0, _0, _0, _0, _0, _0, _0, I9)
    };

    BitTables() noexcept {
        //
    }
};

static const BitTables bit_tables {};

} // namespace jmSudoku

#endif // JM_SUDOKU_TABLES_H
