/*
Copyright 2020 Lucas Heitzmann Gabrielli.
This file is part of gdstk, distributed under the terms of the
Boost Software License - Version 1.0.  See the accompanying
LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>
*/

static PyObject* node_object_str(NodeObject* self) {
    char buffer[GDSTK_PRINT_BUFFER_COUNT];
    snprintf(buffer, COUNT(buffer),
             "Node at layer %" PRIu32 ", datatype %" PRIu32 ", with %" PRIu64 " points",
             get_layer(self->node->tag), get_type(self->node->tag),
             self->node->point_array.count);
    return PyUnicode_FromString(buffer);
}

static void node_object_init() {}

static void node_object_dealloc(NodeObject* self) {
    if (self->node) {
        self->node->clear();
        free_allocation(self->node);
    }
    PyObject_Del(self);
}

static PyObject* node_object_get_points(NodeObject* self, void*) {
    const Array<Vec2>* point_array = &self->node->point_array;
    npy_intp dims[] = {(npy_intp)point_array->count, 2};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Unable to create return array.");
        return NULL;
    }
    double* data = (double*)PyArray_DATA((PyArrayObject*)result);
    memcpy(data, point_array->items, sizeof(double) * point_array->count * 2);
    return (PyObject*)result;
}

static PyObject* node_object_get_layer(NodeObject* self, void*) {
    return PyLong_FromUnsignedLongLong(get_layer(self->node->tag));
}

static PyObject* node_object_get_datatype(NodeObject* self, void*) {
    return PyLong_FromUnsignedLongLong(get_type(self->node->tag));
}

static PyMethodDef node_object_methods[] = {
    {NULL}
};

static PyGetSetDef node_object_getset[] = {
    {"layer", (getter)node_object_get_layer, NULL, label_object_layer_doc, NULL},
    {"datatype", (getter)node_object_get_datatype, NULL, label_object_texttype_doc, NULL},
    {NULL}
};
