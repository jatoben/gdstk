/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

#include "node.h"

namespace gdstk {

void Node::print(bool all) const {
    printf("Node <%p>, count %" PRIu64 ", layer %" PRIu32 ", nodetype %" PRIu32
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

ErrorCode Node::to_gds(FILE* out, double scaling) const {
    printf("Node::to_gds()\n");

    ErrorCode error_code = ErrorCode::NoError;
    uint16_t buffer_start[] = {4,
                               0x1500,
                               6,
                               0x0D02,
                               (uint16_t)get_layer(tag),
                               6,
                               0x2A02,
                               (uint16_t)get_type(tag),
                              };
    big_endian_swap16(buffer_start, COUNT(buffer_start));

    uint16_t buffer_end[] = {4, 0x1100};
    big_endian_swap16(buffer_end, COUNT(buffer_end));

    // TODO points

    fwrite(buffer_start, sizeof(uint16_t), COUNT(buffer_start), out);
    fwrite(buffer_end, sizeof(uint16_t), COUNT(buffer_end), out);
    return error_code;
}

} // namespace gdstk
