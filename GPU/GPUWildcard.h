/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Refactored in 2025 by Zielar
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

namespace GPUWildcard {

/**
 * @brief Matches a string against a wildcard pattern
 * @param str The input string to match
 * @param pattern The wildcard pattern (supports * and ?)
 * @return true if the string matches the pattern, false otherwise
 */
__device__ __noinline__ bool Match(const char* str, const char* pattern) {
    const char* str_ptr;
    const char* pattern_ptr;
    bool star = false;

    while (true) {
        // Reset pointers for new matching attempt
        str_ptr = str;
        pattern_ptr = pattern;

        // Match characters until end of string
        while (*str_ptr) {
            switch (*pattern_ptr) {
                case '?':
                    if (*str_ptr == '.') goto handle_star;
                    break;

                case '*':
                    star = true;
                    str = str_ptr;
                    pattern = pattern_ptr;
                    if (!*++pattern) return true;
                    goto continue_outer_loop;

                default:
                    if (*str_ptr != *pattern_ptr)
                        goto handle_star;
                    break;
            }
            ++str_ptr;
            ++pattern_ptr;
        }

        // Handle trailing star
        if (*pattern_ptr == '*') ++pattern_ptr;
        return (!*pattern_ptr);

    handle_star:
        if (!star) return false;
        ++str;

    continue_outer_loop:
        continue;
    }
}

} // namespace GPUWildcard
