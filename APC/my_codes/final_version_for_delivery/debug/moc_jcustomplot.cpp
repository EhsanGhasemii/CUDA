/****************************************************************************
** Meta object code from reading C++ file 'jcustomplot.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../Plotters/jcustomplot.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'jcustomplot.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_JCustomPlot_t {
    QByteArrayData data[11];
    char stringdata0[82];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_JCustomPlot_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_JCustomPlot_t qt_meta_stringdata_JCustomPlot = {
    {
QT_MOC_LITERAL(0, 0, 11), // "JCustomPlot"
QT_MOC_LITERAL(1, 12, 12), // "changeXEvent"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 4), // "minX"
QT_MOC_LITERAL(4, 31, 4), // "maxX"
QT_MOC_LITERAL(5, 36, 12), // "changeYEvent"
QT_MOC_LITERAL(6, 49, 4), // "minY"
QT_MOC_LITERAL(7, 54, 4), // "maxY"
QT_MOC_LITERAL(8, 59, 7), // "changeX"
QT_MOC_LITERAL(9, 67, 7), // "changeY"
QT_MOC_LITERAL(10, 75, 6) // "ploted"

    },
    "JCustomPlot\0changeXEvent\0\0minX\0maxX\0"
    "changeYEvent\0minY\0maxY\0changeX\0changeY\0"
    "ploted"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_JCustomPlot[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   39,    2, 0x06 /* Public */,
       5,    2,   44,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       8,    2,   49,    2, 0x0a /* Public */,
       9,    2,   54,    2, 0x0a /* Public */,
      10,    0,   59,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    3,    4,
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    6,    7,

 // slots: parameters
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    3,    4,
    QMetaType::Void, QMetaType::Double, QMetaType::Double,    6,    7,
    QMetaType::Void,

       0        // eod
};

void JCustomPlot::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<JCustomPlot *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->changeXEvent((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 1: _t->changeYEvent((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 2: _t->changeX((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 3: _t->changeY((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 4: _t->ploted(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (JCustomPlot::*)(double , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JCustomPlot::changeXEvent)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (JCustomPlot::*)(double , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JCustomPlot::changeYEvent)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject JCustomPlot::staticMetaObject = { {
    QMetaObject::SuperData::link<QCustomPlot::staticMetaObject>(),
    qt_meta_stringdata_JCustomPlot.data,
    qt_meta_data_JCustomPlot,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *JCustomPlot::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *JCustomPlot::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_JCustomPlot.stringdata0))
        return static_cast<void*>(this);
    return QCustomPlot::qt_metacast(_clname);
}

int JCustomPlot::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QCustomPlot::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void JCustomPlot::changeXEvent(double _t1, double _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void JCustomPlot::changeYEvent(double _t1, double _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
