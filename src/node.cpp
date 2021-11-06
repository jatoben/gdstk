/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "node.h"

namespace gdstk {

void Node::print(bool all) const {
    printf("Node <%p>, count %" PRIu64 ", layer %" PRIu32 ", datatype %" PRIu32
           ", owner <%p>\n",
           this, point_array.count, get_layer(tag), get_type(tag), owner);
    if (all) {
        printf("Points: ");
        point_array.print(true);
    }
}

void Node::clear() {
    point_array.clear();
}

// ErrorCode Node::to_gds(FILE* out, double scaling) const {
//     ErrorCode error_code = ErrorCode::NoError;
//     return error_code;
// }

} // namespace gdstk
