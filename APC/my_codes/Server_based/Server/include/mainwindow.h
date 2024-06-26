#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <iostream>
#include <armadillo>
#include <QTime>
#include <QMap>

const int data_size = 3200; 

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE


struct ServerData
{

    ServerData()
    {
        memset(this, 0, sizeof(ServerData));
        header = 0xFA14;
        footer = 0xFB14;
    }
    quint16 header               ;
    double data_I[data_size]         ;
    double data_Q[data_size]         ;
    quint16 range             {0};
    quint16 azimuth           {0};
    quint16 elevation         {0};
    quint16 footer               ;

    quint16 file_num; 
    quint16 chunk_num; 
    quint16 step; 

    quint16 batch_size; 
    quint16 data_num; 
    quint16 offset; 
    quint16 element_len; 
    quint16 out_len; 
    double sigma; 

};

struct ClientData
{

    ClientData()
    {
        memset(this, 0, sizeof(ClientData));
        header = 0xFA15;
        footer = 0xFB15;
    }
    quint16 header               ;
    double data_I[data_size]         ;
    double data_Q[data_size]         ;
    quint16 footer               ;

    quint16 file_num; 
    quint16 chunk_num;
    quint16 step;

    quint16 data_num; 
    quint16 out_len; 

};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QTcpServer             *server                   {nullptr};
    QTimer                 *timer                    {nullptr};
    QList<QTcpSocket*>      clientList                        ;
    ServerData              serverData                        ;

    // Variables related to the main Algorithm ======
    int batch_size; 
    int data_num; 
    int offset; 
    int element_len; 
    int out_len; 
    double sigma; 

    int my_indx; 
    arma::mat y_noisy2_real; 
    arma::mat y_noisy2_imag;
    arma::cx_mat matlab_apc;
    arma::cx_mat gpu_apc;
    // ==============================================
    QMap<int, QTime> messageTimes; 




private slots:
    void clientAppend()     ;
    void clientRemoved()    ;
    void timerTrigged()     ;

};
#endif // MAINWINDOW_H
