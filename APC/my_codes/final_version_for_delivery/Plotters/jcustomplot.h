#ifndef JCUSTOMPLOT_H
#define JCUSTOMPLOT_H

#include <QLabel>
#include <QDebug>
#include <QRubberBand>

#include "qcustomplot.h"


class JCustomPlot : public QCustomPlot
{
    Q_OBJECT
public:
    explicit JCustomPlot(QWidget *parent=0);
    ~JCustomPlot();
    bool zoomEnable() const;
    void setZoomEnable(bool zoomEnable);

    bool zoomMode() const;
    void setZoomMode(bool zoomMode);



    bool ESA() const;
    void setESA(bool ESA);

    bool ClipBound() const;
    void setClipBound(bool ClipBound);

protected:
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void preocessZoomAria(QRect zoomRect);

private:
    QRubberBand *rubberband;
    QPoint zoomStartPos , zoomEndPos;
    bool _zoomEnable;
    bool _zoomMode;
    bool m_ESA;
    bool m_ClipBound;
signals:
    void changeXEvent(double minX , double maxX);
    void changeYEvent(double minY , double maxY);
public slots:
    void changeX(double minX , double maxX);
    void changeY(double minY , double maxY);
    void ploted();

};

#endif // JCUSTOMPLOT_H
