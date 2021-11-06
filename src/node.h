/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#ifndef GDSTK_HEADER_NODE
#define GDSTK_HEADER_NODE

#define __STDC_FORMAT_MACROS
#define _USE_MATH_DEFINES

#include "array.h"
#include "utils.h"

namespace gdstk {

struct Node {
    Tag tag;
    Array<Vec2> point_array;

    // Used by the python interface to store the associated PyObject* (if any).
    // No functions in gdstk namespace should touch this value!
    void *owner;

    void print(bool all) const;

    void clear();

    // ErrorCode to_gds(FILE* out, double scaling) const;
};

} // namespace gdstk

#endif
