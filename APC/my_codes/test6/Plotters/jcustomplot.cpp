#include "jcustomplot.h"

JCustomPlot::JCustomPlot(QWidget *parent)
    :QCustomPlot(parent)
{

    rubberband = new QRubberBand(QRubberBand::Rectangle,this);

    QPalette pal;
    pal.setBrush(QPalette::Highlight, QBrush(QColor("#A0A0A0")));
    rubberband->setPalette(pal);

    _zoomEnable = false;
    connect(this,SIGNAL(beforeReplot()),this,SLOT(ploted()));


    this->setNoAntialiasingOnDrag(true);
    this->setInteractions(QCP::iRangeZoom);
    this->axisRect()->setupFullAxesBox();
    this->setZoomMode(true);
    m_ESA = false;
    m_ClipBound = false;

}

JCustomPlot::~JCustomPlot()
{
}

void JCustomPlot::mouseDoubleClickEvent(QMouseEvent *event)
{
    QCustomPlot::mouseDoubleClickEvent(event);

    if(m_ESA)
    {
        this->xAxis->setRange(143.9,160.1);
        this->xAxis2->setRange(143.9,160.1);
        this->yAxis->setRange(-110,10);
        this->yAxis2->setRange(-110,10);
    }
    else
        rescaleAxes(true);

    this->replot();
}

void JCustomPlot::mousePressEvent(QMouseEvent *event)
{
    QCustomPlot::mousePressEvent(event);
    if(_zoomMode)
    {
        _zoomEnable = true;
        zoomStartPos = event->pos();
        rubberband->setVisible(true);
        rubberband->resize(1,1);
    }
}

void JCustomPlot::mouseReleaseEvent(QMouseEvent *event)
{
    QCustomPlot::mouseReleaseEvent(event);
    if(_zoomMode && _zoomEnable)
    {
        rubberband->setVisible(false);

        zoomEndPos = event->pos();
        QRect rect = QRect(zoomStartPos , zoomEndPos);
        rect = rect.normalized();
        preocessZoomAria(rect);
        rubberband->resize(1,1);
        _zoomEnable = false;
    }
}

void JCustomPlot::mouseMoveEvent(QMouseEvent *event)
{
    QCustomPlot::mouseMoveEvent(event);
    if(_zoomEnable)
    {
        QRect rect = QRect(zoomStartPos , event->pos());
        rect = rect.normalized();
        rubberband->setGeometry(rect);
    }
}

void JCustomPlot::preocessZoomAria(QRect zoomRect)
{
    QRect rect = zoomRect.normalized();

    double x1 = xAxis->pixelToCoord(rect.topLeft().x());
    double x2 = xAxis->pixelToCoord(rect.bottomRight().x());

    double y1 = yAxis->pixelToCoord(rect.topLeft().y());
    double y2 = yAxis->pixelToCoord(rect.bottomRight().y());

    this->xAxis->setRange(x1 , x2);
    this->yAxis->setRange(y1 , y2);

    this->replot();
}

bool JCustomPlot::ClipBound() const
{
    return m_ClipBound;
}

void JCustomPlot::setClipBound(bool ClipBound)
{
    m_ClipBound = ClipBound;
}

bool JCustomPlot::ESA() const
{
    return m_ESA;
}

void JCustomPlot::setESA(bool ESA)
{
    m_ESA = ESA;
}

bool JCustomPlot::zoomMode() const
{
    return _zoomMode;
}

void JCustomPlot::setZoomMode(bool zoomMode)
{
    _zoomMode = zoomMode;
}

bool JCustomPlot::zoomEnable() const
{
    return _zoomEnable;
}

void JCustomPlot::setZoomEnable(bool zoomEnable)
{
    _zoomEnable = zoomEnable;
}

void JCustomPlot::changeX(double minX, double maxX)
{
    if(minX != this->xAxis->range().lower ||
            maxX != this->xAxis->range().upper)
    {
        this->xAxis->setRange(minX , maxX);
        this->replot();
    }
}

void JCustomPlot::changeY(double minY, double maxY)
{
    if(minY != this->yAxis->range().lower ||
            maxY != this->yAxis->range().upper)
    {
        this->yAxis->setRange(minY , maxY);
        this->replot();
    }
}

void JCustomPlot::ploted()
{
    double minx = this->xAxis->range().lower;
    double maxx = this->xAxis->range().upper;
    double miny = this->yAxis->range().lower;
    double maxy = this->yAxis->range().upper;

    if(m_ClipBound && (minx <0 || maxx > 1100))
    {
        minx = 0;
        maxx = 1100;
        this->xAxis->setRange(minx , maxx);
        this->replot();
    }

    emit changeXEvent(minx , maxx);
    emit changeYEvent(miny , maxy);
}


