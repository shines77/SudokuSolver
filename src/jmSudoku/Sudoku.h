
#ifndef JM_SUDOKU_H
#define JM_SUDOKU_H

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

#include "BitSet.h"

using namespace jstd;

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

    const unsigned int box[81] = {
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

    const unsigned int box_X_9[81] = {
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

    const unsigned int box_base[81] = {
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

    const uint8_t cell_info[81][9] = {
        {  0,  0,  0,  0,  0,  0,  0,  0,  0 },  // 0
        {  0,  1,  0,  0,  0,  0,  1,  1,  0 },  // 1
        {  0,  2,  0,  0,  0,  0,  2,  2,  0 },  // 2
        {  0,  3,  1,  3,  1,  1,  0,  0,  0 },  // 3
        {  0,  4,  1,  3,  1,  1,  1,  1,  0 },  // 4
        {  0,  5,  1,  3,  1,  1,  2,  2,  0 },  // 5
        {  0,  6,  2,  6,  2,  2,  0,  0,  0 },  // 6
        {  0,  7,  2,  6,  2,  2,  1,  1,  0 },  // 7
        {  0,  8,  2,  6,  2,  2,  2,  2,  0 },  // 8
        {  1,  0,  0,  0,  0,  0,  3,  0,  1 },  // 9
        {  1,  1,  0,  0,  0,  0,  4,  1,  1 },  // 10
        {  1,  2,  0,  0,  0,  0,  5,  2,  1 },  // 11
        {  1,  3,  1,  3,  1,  1,  3,  0,  1 },  // 12
        {  1,  4,  1,  3,  1,  1,  4,  1,  1 },  // 13
        {  1,  5,  1,  3,  1,  1,  5,  2,  1 },  // 14
        {  1,  6,  2,  6,  2,  2,  3,  0,  1 },  // 15
        {  1,  7,  2,  6,  2,  2,  4,  1,  1 },  // 16
        {  1,  8,  2,  6,  2,  2,  5,  2,  1 },  // 17
        {  2,  0,  0,  0,  0,  0,  6,  0,  2 },  // 18
        {  2,  1,  0,  0,  0,  0,  7,  1,  2 },  // 19
        {  2,  2,  0,  0,  0,  0,  8,  2,  2 },  // 20
        {  2,  3,  1,  3,  1,  1,  6,  0,  2 },  // 21
        {  2,  4,  1,  3,  1,  1,  7,  1,  2 },  // 22
        {  2,  5,  1,  3,  1,  1,  8,  2,  2 },  // 23
        {  2,  6,  2,  6,  2,  2,  6,  0,  2 },  // 24
        {  2,  7,  2,  6,  2,  2,  7,  1,  2 },  // 25
        {  2,  8,  2,  6,  2,  2,  8,  2,  2 },  // 26
        {  3,  0,  3, 27,  0,  0,  0,  0,  0 },  // 27
        {  3,  1,  3, 27,  0,  0,  1,  1,  0 },  // 28
        {  3,  2,  3, 27,  0,  0,  2,  2,  0 },  // 29
        {  3,  3,  4, 30,  1,  1,  0,  0,  0 },  // 30
        {  3,  4,  4, 30,  1,  1,  1,  1,  0 },  // 31
        {  3,  5,  4, 30,  1,  1,  2,  2,  0 },  // 32
        {  3,  6,  5, 33,  2,  2,  0,  0,  0 },  // 33
        {  3,  7,  5, 33,  2,  2,  1,  1,  0 },  // 34
        {  3,  8,  5, 33,  2,  2,  2,  2,  0 },  // 35
        {  4,  0,  3, 27,  0,  0,  3,  0,  1 },  // 36
        {  4,  1,  3, 27,  0,  0,  4,  1,  1 },  // 37
        {  4,  2,  3, 27,  0,  0,  5,  2,  1 },  // 38
        {  4,  3,  4, 30,  1,  1,  3,  0,  1 },  // 39
        {  4,  4,  4, 30,  1,  1,  4,  1,  1 },  // 40
        {  4,  5,  4, 30,  1,  1,  5,  2,  1 },  // 41
        {  4,  6,  5, 33,  2,  2,  3,  0,  1 },  // 42
        {  4,  7,  5, 33,  2,  2,  4,  1,  1 },  // 43
        {  4,  8,  5, 33,  2,  2,  5,  2,  1 },  // 44
        {  5,  0,  3, 27,  0,  0,  6,  0,  2 },  // 45
        {  5,  1,  3, 27,  0,  0,  7,  1,  2 },  // 46
        {  5,  2,  3, 27,  0,  0,  8,  2,  2 },  // 47
        {  5,  3,  4, 30,  1,  1,  6,  0,  2 },  // 48
        {  5,  4,  4, 30,  1,  1,  7,  1,  2 },  // 49
        {  5,  5,  4, 30,  1,  1,  8,  2,  2 },  // 50
        {  5,  6,  5, 33,  2,  2,  6,  0,  2 },  // 51
        {  5,  7,  5, 33,  2,  2,  7,  1,  2 },  // 52
        {  5,  8,  5, 33,  2,  2,  8,  2,  2 },  // 53
        {  6,  0,  6, 54,  0,  0,  0,  0,  0 },  // 54
        {  6,  1,  6, 54,  0,  0,  1,  1,  0 },  // 55
        {  6,  2,  6, 54,  0,  0,  2,  2,  0 },  // 56
        {  6,  3,  7, 57,  1,  1,  0,  0,  0 },  // 57
        {  6,  4,  7, 57,  1,  1,  1,  1,  0 },  // 58
        {  6,  5,  7, 57,  1,  1,  2,  2,  0 },  // 59
        {  6,  6,  8, 60,  2,  2,  0,  0,  0 },  // 60
        {  6,  7,  8, 60,  2,  2,  1,  1,  0 },  // 61
        {  6,  8,  8, 60,  2,  2,  2,  2,  0 },  // 62
        {  7,  0,  6, 54,  0,  0,  3,  0,  1 },  // 63
        {  7,  1,  6, 54,  0,  0,  4,  1,  1 },  // 64
        {  7,  2,  6, 54,  0,  0,  5,  2,  1 },  // 65
        {  7,  3,  7, 57,  1,  1,  3,  0,  1 },  // 66
        {  7,  4,  7, 57,  1,  1,  4,  1,  1 },  // 67
        {  7,  5,  7, 57,  1,  1,  5,  2,  1 },  // 68
        {  7,  6,  8, 60,  2,  2,  3,  0,  1 },  // 69
        {  7,  7,  8, 60,  2,  2,  4,  1,  1 },  // 70
        {  7,  8,  8, 60,  2,  2,  5,  2,  1 },  // 71
        {  8,  0,  6, 54,  0,  0,  6,  0,  2 },  // 72
        {  8,  1,  6, 54,  0,  0,  7,  1,  2 },  // 73
        {  8,  2,  6, 54,  0,  0,  8,  2,  2 },  // 74
        {  8,  3,  7, 57,  1,  1,  6,  0,  2 },  // 75
        {  8,  4,  7, 57,  1,  1,  7,  1,  2 },  // 76
        {  8,  5,  7, 57,  1,  1,  8,  2,  2 },  // 77
        {  8,  6,  8, 60,  2,  2,  6,  0,  2 },  // 78
        {  8,  7,  8, 60,  2,  2,  7,  1,  2 },  // 79
        {  8,  8,  8, 60,  2,  2,  8,  2,  2 }   // 80
    };

    const uint8_t neighbor_cells[81][20] = {
        {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 18, 19, 20, 27, 36, 45, 54, 63, 72 },  // 0
        {  0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 18, 19, 20, 28, 37, 46, 55, 64, 73 },  // 1
        {  0,  1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 18, 19, 20, 29, 38, 47, 56, 65, 74 },  // 2
        {  0,  1,  2,  4,  5,  6,  7,  8, 12, 12, 13, 14, 21, 21, 30, 39, 48, 57, 66, 75 },  // 3
        {  0,  1,  2,  3,  5,  6,  7,  8, 12, 13, 13, 14, 21, 22, 31, 40, 49, 58, 67, 76 },  // 4
        {  0,  1,  2,  3,  4,  6,  7,  8, 12, 13, 14, 14, 21, 23, 32, 41, 50, 59, 68, 77 },  // 5
        {  0,  1,  2,  3,  4,  5,  7,  8, 15, 15, 16, 17, 24, 24, 33, 42, 51, 60, 69, 78 },  // 6
        {  0,  1,  2,  3,  4,  5,  6,  8, 15, 16, 16, 17, 24, 25, 34, 43, 52, 61, 70, 79 },  // 7
        {  0,  1,  2,  3,  4,  5,  6,  7, 15, 16, 17, 17, 24, 26, 35, 44, 53, 62, 71, 80 },  // 8
        {  0,  1,  2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 36, 45, 54, 63, 72 },  // 9
        {  0,  1,  2,  9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 28, 37, 46, 55, 64, 73 },  // 10
        {  0,  1,  2,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 29, 38, 47, 56, 65, 74 },  // 11
        {  3,  3,  4,  5,  9, 10, 11, 13, 14, 15, 16, 17, 21, 21, 30, 39, 48, 57, 66, 75 },  // 12
        {  3,  4,  4,  5,  9, 10, 11, 12, 14, 15, 16, 17, 21, 22, 31, 40, 49, 58, 67, 76 },  // 13
        {  3,  4,  5,  5,  9, 10, 11, 12, 13, 15, 16, 17, 21, 23, 32, 41, 50, 59, 68, 77 },  // 14
        {  6,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 24, 24, 33, 42, 51, 60, 69, 78 },  // 15
        {  6,  7,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17, 24, 25, 34, 43, 52, 61, 70, 79 },  // 16
        {  6,  7,  8,  8,  9, 10, 11, 12, 13, 14, 15, 16, 24, 26, 35, 44, 53, 62, 71, 80 },  // 17
        {  0,  1,  2,  9, 10, 11, 19, 20, 21, 22, 23, 24, 25, 26, 27, 36, 45, 54, 63, 72 },  // 18
        {  0,  1,  2,  9, 10, 11, 18, 20, 21, 22, 23, 24, 25, 26, 28, 37, 46, 55, 64, 73 },  // 19
        {  0,  1,  2,  9, 10, 11, 18, 19, 21, 22, 23, 24, 25, 26, 29, 38, 47, 56, 65, 74 },  // 20
        {  3,  3,  4,  5, 12, 12, 18, 19, 20, 22, 23, 24, 25, 26, 30, 39, 48, 57, 66, 75 },  // 21
        {  3,  4,  4,  5, 12, 13, 18, 19, 20, 21, 23, 24, 25, 26, 31, 40, 49, 58, 67, 76 },  // 22
        {  3,  4,  5,  5, 12, 14, 18, 19, 20, 21, 22, 24, 25, 26, 32, 41, 50, 59, 68, 77 },  // 23
        {  6,  6,  7,  8, 15, 15, 18, 19, 20, 21, 22, 23, 25, 26, 33, 42, 51, 60, 69, 78 },  // 24
        {  6,  7,  7,  8, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 34, 43, 52, 61, 70, 79 },  // 25
        {  6,  7,  8,  8, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 35, 44, 53, 62, 71, 80 },  // 26
        {  0,  9, 18, 28, 28, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 54, 63, 72 },  // 27
        {  1, 10, 19, 27, 27, 29, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 55, 64, 73 },  // 28
        {  2, 11, 20, 27, 27, 28, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 47, 56, 65, 74 },  // 29
        {  3, 12, 21, 27, 28, 29, 31, 31, 32, 32, 33, 34, 35, 39, 39, 40, 48, 57, 66, 75 },  // 30
        {  4, 13, 22, 27, 28, 29, 30, 30, 32, 32, 33, 34, 35, 39, 40, 40, 49, 58, 67, 76 },  // 31
        {  5, 14, 23, 27, 28, 29, 30, 30, 31, 31, 33, 34, 35, 39, 40, 41, 50, 59, 68, 77 },  // 32
        {  6, 15, 24, 27, 28, 29, 30, 31, 32, 34, 34, 35, 35, 42, 42, 43, 51, 60, 69, 78 },  // 33
        {  7, 16, 25, 27, 28, 29, 30, 31, 32, 33, 33, 35, 35, 42, 43, 43, 52, 61, 70, 79 },  // 34
        {  8, 17, 26, 27, 28, 29, 30, 31, 32, 33, 33, 34, 34, 42, 43, 44, 53, 62, 71, 80 },  // 35
        {  0,  9, 18, 27, 28, 29, 37, 37, 38, 38, 39, 40, 41, 42, 43, 44, 45, 54, 63, 72 },  // 36
        {  1, 10, 19, 27, 28, 29, 36, 36, 38, 38, 39, 40, 41, 42, 43, 44, 46, 55, 64, 73 },  // 37
        {  2, 11, 20, 27, 28, 29, 36, 36, 37, 37, 39, 40, 41, 42, 43, 44, 47, 56, 65, 74 },  // 38
        {  3, 12, 21, 30, 30, 31, 32, 36, 37, 38, 40, 40, 41, 42, 43, 44, 48, 57, 66, 75 },  // 39
        {  4, 13, 22, 30, 31, 31, 32, 36, 37, 38, 39, 39, 41, 42, 43, 44, 49, 58, 67, 76 },  // 40
        {  5, 14, 23, 30, 31, 32, 32, 36, 37, 38, 39, 39, 40, 42, 43, 44, 50, 59, 68, 77 },  // 41
        {  6, 15, 24, 33, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 43, 44, 51, 60, 69, 78 },  // 42
        {  7, 16, 25, 33, 34, 34, 35, 36, 37, 38, 39, 40, 41, 42, 42, 44, 52, 61, 70, 79 },  // 43
        {  8, 17, 26, 33, 34, 35, 35, 36, 37, 38, 39, 40, 41, 42, 42, 43, 53, 62, 71, 80 },  // 44
        {  0,  9, 18, 27, 28, 29, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53, 54, 63, 72 },  // 45
        {  1, 10, 19, 27, 28, 29, 36, 37, 38, 45, 47, 48, 49, 50, 51, 52, 53, 55, 64, 73 },  // 46
        {  2, 11, 20, 27, 28, 29, 36, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 65, 74 },  // 47
        {  3, 12, 21, 30, 30, 31, 32, 39, 39, 45, 46, 47, 49, 50, 51, 52, 53, 57, 66, 75 },  // 48
        {  4, 13, 22, 30, 31, 31, 32, 39, 40, 45, 46, 47, 48, 50, 51, 52, 53, 58, 67, 76 },  // 49
        {  5, 14, 23, 30, 31, 32, 32, 39, 41, 45, 46, 47, 48, 49, 51, 52, 53, 59, 68, 77 },  // 50
        {  6, 15, 24, 33, 33, 34, 35, 42, 42, 45, 46, 47, 48, 49, 50, 52, 53, 60, 69, 78 },  // 51
        {  7, 16, 25, 33, 34, 34, 35, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 61, 70, 79 },  // 52
        {  8, 17, 26, 33, 34, 35, 35, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 62, 71, 80 },  // 53
        {  0,  9, 18, 27, 36, 45, 55, 55, 56, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 72 },  // 54
        {  1, 10, 19, 28, 37, 46, 54, 54, 56, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 73 },  // 55
        {  2, 11, 20, 29, 38, 47, 54, 54, 55, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 74 },  // 56
        {  3, 12, 21, 30, 39, 48, 54, 55, 56, 58, 58, 59, 59, 60, 61, 62, 66, 66, 67, 75 },  // 57
        {  4, 13, 22, 31, 40, 49, 54, 55, 56, 57, 57, 59, 59, 60, 61, 62, 66, 67, 67, 76 },  // 58
        {  5, 14, 23, 32, 41, 50, 54, 55, 56, 57, 57, 58, 58, 60, 61, 62, 66, 67, 68, 77 },  // 59
        {  6, 15, 24, 33, 42, 51, 54, 55, 56, 57, 58, 59, 61, 61, 62, 62, 69, 69, 70, 78 },  // 60
        {  7, 16, 25, 34, 43, 52, 54, 55, 56, 57, 58, 59, 60, 60, 62, 62, 69, 70, 70, 79 },  // 61
        {  8, 17, 26, 35, 44, 53, 54, 55, 56, 57, 58, 59, 60, 60, 61, 61, 69, 70, 71, 80 },  // 62
        {  0,  9, 18, 27, 36, 45, 54, 55, 56, 64, 64, 65, 65, 66, 67, 68, 69, 70, 71, 72 },  // 63
        {  1, 10, 19, 28, 37, 46, 54, 55, 56, 63, 63, 65, 65, 66, 67, 68, 69, 70, 71, 73 },  // 64
        {  2, 11, 20, 29, 38, 47, 54, 55, 56, 63, 63, 64, 64, 66, 67, 68, 69, 70, 71, 74 },  // 65
        {  3, 12, 21, 30, 39, 48, 57, 57, 58, 59, 63, 64, 65, 67, 67, 68, 69, 70, 71, 75 },  // 66
        {  4, 13, 22, 31, 40, 49, 57, 58, 58, 59, 63, 64, 65, 66, 66, 68, 69, 70, 71, 76 },  // 67
        {  5, 14, 23, 32, 41, 50, 57, 58, 59, 59, 63, 64, 65, 66, 66, 67, 69, 70, 71, 77 },  // 68
        {  6, 15, 24, 33, 42, 51, 60, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 70, 71, 78 },  // 69
        {  7, 16, 25, 34, 43, 52, 60, 61, 61, 62, 63, 64, 65, 66, 67, 68, 69, 69, 71, 79 },  // 70
        {  8, 17, 26, 35, 44, 53, 60, 61, 62, 62, 63, 64, 65, 66, 67, 68, 69, 69, 70, 80 },  // 71
        {  0,  9, 18, 27, 36, 45, 54, 55, 56, 63, 64, 65, 73, 74, 75, 76, 77, 78, 79, 80 },  // 72
        {  1, 10, 19, 28, 37, 46, 54, 55, 56, 63, 64, 65, 72, 74, 75, 76, 77, 78, 79, 80 },  // 73
        {  2, 11, 20, 29, 38, 47, 54, 55, 56, 63, 64, 65, 72, 73, 75, 76, 77, 78, 79, 80 },  // 74
        {  3, 12, 21, 30, 39, 48, 57, 57, 58, 59, 66, 66, 72, 73, 74, 76, 77, 78, 79, 80 },  // 75
        {  4, 13, 22, 31, 40, 49, 57, 58, 58, 59, 66, 67, 72, 73, 74, 75, 77, 78, 79, 80 },  // 76
        {  5, 14, 23, 32, 41, 50, 57, 58, 59, 59, 66, 68, 72, 73, 74, 75, 76, 78, 79, 80 },  // 77
        {  6, 15, 24, 33, 42, 51, 60, 60, 61, 62, 69, 69, 72, 73, 74, 75, 76, 77, 79, 80 },  // 78
        {  7, 16, 25, 34, 43, 52, 60, 61, 61, 62, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80 },  // 79
        {  8, 17, 26, 35, 44, 53, 60, 61, 62, 62, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79 }   // 80
    };

    Tables() noexcept {
        //
    }
};

static const Tables tables {};

template <size_t TotalSize>
struct SudokuBoard {
    char board[TotalSize];
};

template <size_t nBoxCellsX = 3, size_t nBoxCellsY = 3,
          size_t nBoxCountX = 3, size_t nBoxCountY = 3,
          size_t nMinNumber = 1, size_t nMaxNumber = 9>
struct BasicSudoku {
    static const size_t kAlignment = sizeof(size_t);

    static const size_t BoxCellsX = nBoxCellsX;     // 3
    static const size_t BoxCellsY = nBoxCellsY;     // 3
    static const size_t BoxCountX = nBoxCountX;     // 3
    static const size_t BoxCountY = nBoxCountY;     // 3
    static const size_t MinNumber = nMinNumber;     // 1
    static const size_t MaxNumber = nMaxNumber;     // 9

    static const size_t Cols = BoxCellsX * BoxCountX;
    static const size_t Rows = BoxCellsY * BoxCountY;
    static const size_t Boxes = BoxCountX * BoxCountY;
    static const size_t Numbers = (MaxNumber - MinNumber) + 1;

    static const size_t BoxSize = BoxCellsX * BoxCellsY;
    static const size_t BoardSize = Rows * Cols;
    static const size_t TotalSize = BoxSize * Boxes * Numbers;

    static const size_t Neighbors = (Cols - 1) + (Rows - 1) +
                                    (BoxSize - (BoxCellsX - 1) - (BoxCellsY - 1) - 1);

    static const size_t MaxEffectBox = (BoxCountX - 1) + (BoxCountY -1) + 1;
    static const size_t MaxEffectLength = MaxEffectBox * BoxSize;

    static const size_t TotalSize2 = Rows * Cols * Numbers;

    static const size_t TotalCellLiterals = Rows * Cols;
    static const size_t TotalRowLiterals = Rows * Numbers;
    static const size_t TotalColLiterals = Cols * Numbers;
    static const size_t TotalBoxLiterals = Boxes * Numbers;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t RowLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t BoxLiteralFirst  = ColLiteralFirst + TotalColLiterals;
    static const size_t LiteralLast      = BoxLiteralFirst + TotalBoxLiterals;

    static const size_t kAllRowsBit = (size_t(1) << Rows) - 1;
    static const size_t kAllColsBit = (size_t(1) << Cols) - 1;
    static const size_t kAllBoxesBit = (size_t(1) << Boxes) - 1;
    static const size_t kAllNumbersBit = (size_t(1) << Numbers) - 1;

#pragma pack(push, 1)

    static const size_t NeighborsAlignBytes = ((Neighbors * sizeof(uint8_t) + kAlignment - 1) / kAlignment) * kAlignment;

    // Aligned to sizeof(size_t) for cache friendly
    struct NeighborCells {
        uint8_t cells[Neighbors];
        uint8_t reserve[NeighborsAlignBytes - Neighbors * sizeof(uint8_t)];
    };

    struct CellInfo {
        uint8_t row, col;
        uint8_t box, box_base;
        uint8_t box_x, box_y;
        uint8_t cell;
        uint8_t cell_x, cell_y;
        // Reserve for cache friendly
        uint8_t reserve[7];
    };

#pragma pack(pop)

    static bool is_inited;

    static CellInfo * cell_info;
    static NeighborCells * neighbor_cells;

    static void initialize() {
        if (!is_inited) {
            product_cell_info();
            product_neighbor_cells();
            is_inited = true;
        }
    }

    static void finalize() {
        if (is_inited) {
            if (cell_info) {
                delete[] cell_info;
                cell_info = nullptr;
            }
            if (neighbor_cells) {
                delete[] neighbor_cells;
                neighbor_cells = nullptr;
            }
            is_inited = false;
        }
    }

    static void product_cell_info() {
        if (cell_info == nullptr) {
            cell_info = new CellInfo[BoardSize];

            size_t pos = 0;
            for (size_t row = 0; row < Rows; row++) {
                for (size_t col = 0; col < Cols; col++) {
                    CellInfo * cellInfo = &cell_info[pos];
                    size_t box_x = col / BoxCellsX;
                    size_t box_y = row / BoxCellsY;
                    size_t box = box_y * BoxCountX + box_x;
                    size_t box_base = (box_y * BoxCellsY) * Cols + box_x * BoxCellsX;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell_y = row % BoxCellsY;
                    size_t cell = cell_y * BoxCellsX + cell_x;

                    cellInfo->row = (uint8_t)row;
                    cellInfo->col = (uint8_t)col;
                    cellInfo->box = (uint8_t)box;
                    cellInfo->box_base = (uint8_t)box_base;
                    cellInfo->box_x = (uint8_t)box_x;
                    cellInfo->box_y = (uint8_t)box_x;
                    cellInfo->cell = (uint8_t)cell;
                    cellInfo->cell_x = (uint8_t)cell_x;
                    cellInfo->cell_y = (uint8_t)cell_y;

                    pos++;
                }
            }

            //print_cell_info();
        }
    }

    static void print_cell_info() {
        printf("const uint8_t cell_info[%d][9] = {\n", (int)BoardSize);
        for (size_t pos = 0; pos < BoardSize; pos++) {
            printf("    {");
            const CellInfo & cellInfo = cell_info[pos];
            printf(" %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d ",
                   (int)cellInfo.row, (int)cellInfo.col,
                   (int)cellInfo.box, (int)cellInfo.box_base,
                   (int)cellInfo.box_x, (int)cellInfo.box_y,
                   (int)cellInfo.cell,
                   (int)cellInfo.cell_x, (int)cellInfo.cell_y);
            if (pos < (BoardSize - 1))
                printf("},  // %d\n", (int)pos);
            else
                printf("}   // %d\n", (int)pos);
        }
        printf("};\n");
    }

    static size_t get_neighbor_cells_list(size_t row, size_t col,
                                          NeighborCells * list) {
        assert(list != nullptr);
        size_t index = 0;
        size_t pos_y = row * Cols;
        for (size_t x = 0; x < Cols; x++) {
            if (x != col) {
                list->cells[index++] = (uint8_t)(pos_y + x);
            }
        }

        size_t pos_x = col;
        for (size_t y = 0; y < Rows; y++) {
            if (y != row) {
                list->cells[index++] = (uint8_t)(y * Cols + pos_x);
            }
        }

        size_t box_x = col / BoxCellsX;
        size_t box_y = row / BoxCellsY;
        size_t box_base = (box_y * BoxCellsY) * Cols + box_x * BoxCellsX;
        size_t pos = pos_y + pos_x;
        size_t cell_x = col % BoxCellsX;
        size_t cell_y = row % BoxCellsY;
        size_t cell = box_base;
        for (size_t y = 0; y < BoxCellsY; y++) {
            if (y == cell_y) {
                cell += Cols;
            }
            else {
                for (size_t x = 0; x < BoxCellsX; x++) {
                    if (x != cell_x) {
                        assert(cell != pos);
                        list->cells[index++] = (uint8_t)(cell);
                    }
                    cell++;
                }
                cell += (Cols - BoxCellsX);
            }
        }

        assert(index == Neighbors);
        return index;
    }

    static void product_neighbor_cells() {
        if (neighbor_cells == nullptr) {
            neighbor_cells = new NeighborCells[BoardSize];

            size_t pos = 0;
            for (size_t row = 0; row < Rows; row++) {
                for (size_t col = 0; col < Cols; col++) {
                    NeighborCells * list = &neighbor_cells[pos];
                    size_t neighbors = get_neighbor_cells_list(row, col, list);
                    assert(neighbors == Neighbors);
                    // Sort the cells for cache friendly
                    std::sort(&neighbor_cells[pos].cells[0], &neighbor_cells[pos].cells[Neighbors]);
                    pos++;
                }
            }

            // print_neighbor_cells();
        }
    }

    static void print_neighbor_cells() {
        printf("const uint8_t neighbor_cells[%d][%d] = {\n", (int)BoardSize, (int)Neighbors);
        for (size_t pos = 0; pos < BoardSize; pos++) {
            printf("    { ");
            for (size_t cell = 0; cell < Neighbors; cell++) {
                if (cell < Neighbors - 1)
                    printf("%2u, ", (uint32_t)neighbor_cells[pos].cells[cell]);
                else
                    printf("%2u ", (uint32_t)neighbor_cells[pos].cells[cell]);
            }
            if (pos < (BoardSize - 1))
                printf("},  // %d\n", (int)pos);
            else
                printf("}   // %d\n", (int)pos);
        }
        printf("};\n");
    }

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
                char val = board[pos++];
                if ((val != '.') && (val != ' ') && (val != '0') && (val != '-')) {
                    filled++;
                }
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
        for (size_t countX = 0; countX < BoxCountX; countX++) {
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
                if ((col % BoxCellsX) == (BoxCellsX - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % BoxCellsY) == (BoxCellsY - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < BoxCountX; countX++) {
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
                char val = line[col];
                if ((val != '.') && (val != ' ') && (val != '0') && (val != '-')) {
                    filled++;
                }
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
        for (size_t countX = 0; countX < BoxCountX; countX++) {
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
                if ((col % BoxCellsX) == (BoxCellsX - 1))
                    printf("| ");
            }
            printf("\n");
            if ((row % BoxCellsY) == (BoxCellsY - 1)) {
                // printf("  ------- ------- -------\n");
                printf(" ");
                for (size_t countX = 0; countX < BoxCountX; countX++) {
                    printf(" -------");
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    static void display_boards(std::vector<SudokuBoard<BasicSudoku::BoardSize>> & boards) {
        printf("Total answers: %d\n\n", (int)boards.size());
        int i = 0;
        for (auto answer : boards) {
            BasicSudoku::display_board(answer.board, false, i);
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

template <size_t nBoxCellsX, size_t nBoxCellsY,
          size_t nBoxCountX, size_t nBoxCountY,
          size_t nMinNumber, size_t nMaxNumber>
bool BasicSudoku<nBoxCellsX, nBoxCellsY,
                 nBoxCountX, nBoxCountY,
                 nMinNumber, nMaxNumber>::is_inited = false;

template <size_t nBoxCellsX, size_t nBoxCellsY,
          size_t nBoxCountX, size_t nBoxCountY,
          size_t nMinNumber, size_t nMaxNumber>
typename BasicSudoku<nBoxCellsX, nBoxCellsY,
                     nBoxCountX, nBoxCountY,
                     nMinNumber, nMaxNumber>::CellInfo *
    BasicSudoku<nBoxCellsX, nBoxCellsY,
                nBoxCountX, nBoxCountY,
                nMinNumber, nMaxNumber>::cell_info = nullptr;

template <size_t nBoxCellsX, size_t nBoxCellsY,
          size_t nBoxCountX, size_t nBoxCountY,
          size_t nMinNumber, size_t nMaxNumber>
typename BasicSudoku<nBoxCellsX, nBoxCellsY,
                     nBoxCountX, nBoxCountY,
                     nMinNumber, nMaxNumber>::NeighborCells *
    BasicSudoku<nBoxCellsX, nBoxCellsY,
                nBoxCountX, nBoxCountY,
                nMinNumber, nMaxNumber>::neighbor_cells = nullptr;

// Standard sudoku definition
typedef BasicSudoku<3, 3, 3, 3, 1, 9> Sudoku;

} // namespace jmSudoku

#endif // JM_SUDOKU_H
