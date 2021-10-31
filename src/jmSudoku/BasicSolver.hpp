
#include "BasicSolver.h"

template <typename SudokuTy>
size_t jmSudoku::BasicSolver<SudokuTy>::num_guesses = 0;

template <typename SudokuTy>
size_t jmSudoku::BasicSolver<SudokuTy>::num_unique_candidate = 0;

template <typename SudokuTy>
size_t jmSudoku::BasicSolver<SudokuTy>::num_failed_return = 0;
