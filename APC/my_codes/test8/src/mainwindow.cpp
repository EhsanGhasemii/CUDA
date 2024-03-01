#include "../include/mainwindow.h"
#include "../include/ui_mainwindow.h"

#include "../include/matchfilter.h"
#include "Plotters/qcustomplot.h"

// additional libraries
//#include "cuda_main.h"

void my_fill(double* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = 0;
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    int temp=0;
    s.resize(13,1);
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;
    s(temp++,0)=-1;
    s(temp++,0)=1;



	////
	alpha = mat(1, 1); 
	alpha(0,0) = 1.9;
	//alpha(1,0) = 1.8;



    plotter = new QCustomPlot;
    plotter->addGraph();
    plotter->addGraph();
    plotter->addGraph();

    QVBoxLayout *lay = new QVBoxLayout;
    lay->addWidget(plotter);
    ui->widget->setLayout(lay);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    QString str = QFileDialog::getOpenFileName(nullptr,"Select I File");
    if(!str.isEmpty())
        File_I = new QFile(str);
    str = QFileDialog::getOpenFileName(nullptr,"Select Q File");
    if(!str.isEmpty())
        File_Q = new QFile(str);

    File_I->open(QIODevice::ReadOnly);
    File_Q->open(QIODevice::ReadOnly);


	// start y_noisy collection from 0
	my_indx = 0; 


	// Get the starting timepoint
    auto start1 = std::chrono::high_resolution_clock::now();


    if(File_I != nullptr) {

        if(File_I ->isOpen() && File_Q->isOpen()) {

			// modifying ============================
            while (!File_I->atEnd() && !File_Q->atEnd()) {
			// ======================================

            QString str_I = File_I->readLine();
            QString str_Q = File_Q->readLine();

            QStringList ls_I = str_I.split(',');
            QStringList ls_Q = str_Q.split(',');

            if(ls_I.count() == ls_Q.count())
            {
                y_noisy.resize(ls_I.count());
                for (int var =0; var < ls_I.count(); ++var) {
                    y_noisy.at(var) = cx_double(ls_I.at(var).toDouble(),ls_Q.at(var).toDouble());
                }
				y_noisy2.push_back(y_noisy); 

            }		// if(ls_I.count() == ls_Q.count())

			// modifying =======================================
			}		// while
			// =================================================
        }			// if(File_I ->isOpen() && File_Q->isOpen()) {
    }				// if(File_I != nullptr) {


	// Get the ending timepoint
    auto stop1 = std::chrono::high_resolution_clock::now();

	// calculate time processing
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
	std::cout << "Reading duration time: " << duration1.count() << " microseconds" << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl; 



	// start coding CUDA =============================
	
	// allocate memory in CPU for calculation
	double* X_real; 
	double* X_imag; 
	double* R_real; 
	double* R_imag;
	double* Ss_real; 
	double* Ss_imag; 
	double* s_real; 
	double* s_imag;
	double* alpha_real; 

	double* output; 



	// prepare our GPU format data
	// convert cx_mat of armadillo library to ordinary arrays. 
	fun1(s, 
		 y_noisy2[0], 
		 13, 
		 alpha, 
		 1e-5, 
		 &X_real, 
		 &X_imag, 
		 &R_real, 
		 &R_imag,
		 &Ss_real,
		 &Ss_imag,
		 &s_real,
		 &s_imag,
		 &alpha_real,
		 &output
		 );


	// main GPU kernel
	gpuKernel(X_real,
			  X_imag,
			  R_real,
			  R_imag,
			  Ss_real,
			  Ss_imag,
			  s_real,
			  s_imag, 
			  alpha_real,
			  output
			  );

	// free the memory that we use
	free(X_real); 
	free(X_imag);
	free(R_real); 
	free(R_imag);
	free(Ss_real); 
	free(Ss_imag); 
	free(s_real); 
	free(s_imag); 
	free(alpha_real); 


	// modifying =================================
	cx_mat help = zeros<cx_mat>(3, 3);
	help(0, 0) = std::complex<double>(1, 1); 
	help(0, 1) = 1; 
	help(0, 2) = 1; 
	help(1, 0) = 2; 
	help(1, 1) = 4; 
	help(1, 2) = 6; 
	help(2, 0) = 3; 
	help(2, 1) = 6; 
	help(2, 2) = 12; 

	std::cout << " 555555555555555555555 " << std::endl; 
	std::cout << "help : " << std::endl;						// size: 13 * 13
	for(int i=0; i<help.n_rows; i++){
		for(int j=0; j<help.n_cols; j++){
			std::cout << "help(" << i << ", " << j << "): ";
			std::cout << help(i,j) << "\t";
		}
		std::cout << std::endl;
	}
	cx_mat my_inv = inv(help); 
	std::cout << "my_inv : " << std::endl;						// size: 13 * 13
	for(int i=0; i<my_inv.n_rows; i++){
		for(int j=0; j<my_inv.n_cols; j++){
			std::cout << "my_inv(" << i << ", " << j << "): ";
			std::cout << my_inv(i,j) << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << " 555555555555555555555 " << std::endl;
	// ===========================================



    readLine();
}

void MainWindow::on_pushButton_2_clicked()
{
	my_indx ++; 
    readLine();
}


void MainWindow::readLine()
{
	// Get the ending timepoint
    auto start = std::chrono::high_resolution_clock::now();

	std::cout << "indx: " << my_indx << std::endl;
	std::cout << "y_noisy size: " << y_noisy2.size() << std::endl; 
	// exclude our target y_noisy
	y_noisy = y_noisy2[my_indx]; 


   


				// modifying ====================================
				/*
				std::cout << " ------ " << std::endl; 
				std::cout << "y noisy : " << std::endl; 
				for(int i=0; i<y_noisy.n_rows; i++){
					for(int j=0; j<y_noisy.n_cols; j++){
						std::cout << "y_noisy(" << i << ", " << j << "): ";
						std::cout << y_noisy(i,j) << "\t";
					}
					std::cout << std::endl;
				}*/

				// =============================================
				



                General_APC apc;
                MatchFilter mf;

                cx_mat result_apc = apc.algorithm(s, y_noisy, 13, alpha, 1e-5);
                cx_mat result_mf = mf.algorithm(s, y_noisy, 13);


                cx_mat result_apc2 = apc.algorithm2(s, y_noisy, 13, alpha, 1e-5);
                cx_mat result_mf2 = mf.algorithm(s, y_noisy, 13);

				//std::cout << "size of result_apc: " << result_apc.size() << std::endl;
				//std::cout << "size of result_mf: " << result_mf.size() << std::endl; 
				



				// Check if the two matrices are equal
				if(arma::approx_equal(result_apc, result_apc2, "absdiff", 0.0001)) {
					std::cout << "The matrices are equal." << std::endl;
				} else {
					std::cout << "The matrices are not equal." << std::endl;
				}





				// modifying ==================================
				
				/*std::cout << "result apc : " << std::endl; 
				for(int i=0; i<result_apc.n_rows; i++){
					for(int j=0; j<result_apc.n_cols; j++){
						std::cout << "result_apc(" << i << ", " << j << "): ";
						std::cout << result_apc(i,j) << "\t";
					}
					std::cout << std::endl;
				}*/

				std::cout << "=========================" << std::endl;
				// ============================================



                for (int var = 0; var < y_noisy.size(); ++var) {
                    //qDebug()<<var<<"==>"<<y_noisy.at(var).real()<<y_noisy.at(var).imag();
                }


                double gt[result_apc.size()];
                for (int var = 0; var < result_apc.size(); ++var) {
                    gt[var] = -80;
                }
                //20dB, 1m/s
                //gt[42]=-1.5;gt[77]=-20;gt[87]=0;

                //20dB, 10m/s
                //gt[35]=-1.5;gt[80]=-19.9;gt[90]=0;

                //20dB, 300m/s
                //gt[35]=-1.5;gt[80]=-19.9;gt[90]=0;//8

                //20dB, 600m/s
                //gt[35]=-1.5;gt[80]=-20;gt[90]=0;//5

                //30dB, 1m/s
                //gt[42]=-1.5;gt[80]=-29;gt[90]=0;

                //30dB, 10m/s
                //gt[35]=-1.5;gt[80]=-29;gt[90]=0;//4

                //30dB, 300m/s
                //gt[35]=-1.5;gt[80]=-29;gt[90]=0;//5

                //30dB, 600m/s
                //gt[35]=-1.5;gt[80]=-29;gt[90]=0;//5

                //50dB, 1m/s
                //gt[42]=-1.5;gt[80]=-49;gt[90]=0;

                //50dB, 10m/s
                //gt[35]=-1.5;gt[80]=-49;gt[90]=0;

                //50dB, 300m/s
                //gt[35]=-1.5;gt[80]=-49;gt[90]=0;//4

                //50dB, 600m/s
                gt[35]=-1.5;gt[80]=-49;gt[90]=0;//2



                plotter->graph(0)->data()->clear();
                plotter->graph(1)->data()->clear();
                plotter->graph(2)->data()->clear();

                double max = abs(result_apc.max());
                for (int var = 0; var < result_apc.size(); ++var) {
                    plotter->graph(0)->addData(var,20*log10(qAbs(abs(result_apc(var,0))/max)));
                }
                max = abs(result_mf.max());
                for (int var = 0; var < result_mf.size(); ++var) {
                    plotter->graph(1)->addData(var,20*log10(qAbs(abs(result_mf(var,0))/max)));
                }

                for (int var = 0; var < result_mf.size(); ++var) {

                    plotter->graph(2)->addData(var+1,gt[var]);


                }

                plotter->graph(0)->setName("APC");
                plotter->graph(1)->setName("MF");
                plotter->graph(2)->setName("GT");
                plotter->graph(1)->setPen(QPen(Qt::red,1,Qt::DashLine));
                plotter->graph(2)->setPen(QPen(Qt::black,1.5));
                plotter->legend->setVisible(true);
                plotter->xAxis->setLabel("Range cell index");
                plotter->yAxis->setLabel("Power(dB)");
                plotter->setInteraction(QCP::iRangeDrag,true);
                plotter->setInteraction(QCP::iRangeZoom,true);
                plotter->rescaleAxes();
                plotter->yAxis->setRange(-80,0);
                plotter->xAxis->setRange(0,150);
                plotter->replot();
                plotter->show();
                //




	
	// Get the ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();


	// calculate time processing
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Big Time: " << duration.count() << " microseconds" << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl; 
	


}					// void MainWindow::readLine()

