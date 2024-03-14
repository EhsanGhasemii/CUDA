#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QFile>
#include <QMainWindow>
#include <Plotters/qcustomplot.h>
#include "general_apc.h"
#include "cuda_main.h"
#include "gpuFunctions.h"


double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size);



QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

private:
    Ui::MainWindow *ui;
    QFile *File_I,*File_Q;

    QCustomPlot *plotter;
    cx_mat y_noisy,s;

	// modifying ======================
	cx_mat y_noisy2;
	cx_mat gpu_apc; 
	int my_indx;
	mat alpha; 
	// ================================



    void readLine();
};
#endif // MAINWINDOW_H
