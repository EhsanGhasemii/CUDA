#include "../include/mainwindow.h"
#include "../include/ui_mainwindow.h"

#include "../include/matchfilter.h"
#include "Plotters/qcustomplot.h"


void my_fill(double* data, int size) {
    for (int i=0; i<size; ++i)
        data[i] = 0;
}



// function2 for MSE calculation between an array and a cx_mat
double calc_mse (cx_mat data1, cx_mat data2, int row, int col_size) {
	/* We suppose that size of data1 is n * m and
	 * sise of data 2 is 1 * m_prime and we just 
	 * compaire first col_size of (row)th row of 
	 * first data with second data. 
	 */
	double mse = 0.0; 
	for (int i = 0; i < col_size; ++i) {
		mse += (data1(row, i).real() - data2(0, i).real()) * (data1(row, i).real() - data2(0, i).real());
		mse += (data1(row, i).imag() - data2(0, i).imag()) * (data1(row, i).imag() - data2(0, i).imag());
	}
	return mse; 
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
	alpha = mat(2, 1); 
	alpha(0,0) = 1.9;
	alpha(1,0) = 1.8;
	//alpha(2,0) = 1.7; 



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

            while (!File_I->atEnd() && !File_Q->atEnd()) {

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

					// append y_noisy in y_noisy2
					if(y_noisy2.n_elem == 0) {
						y_noisy2 = y_noisy;
					}
					else {
						y_noisy2 = arma::join_rows(y_noisy2, y_noisy);
					}

				}		// if(ls_I.count() == ls_Q.count())
			}		// while
        }			// if(File_I ->isOpen() && File_Q->isOpen()) {
    }				// if(File_I != nullptr) {


	// Get the ending timepoint
    auto stop1 = std::chrono::high_resolution_clock::now();


	std::vector<arma::cx_mat> my_vec;
	my_vec.push_back(y_noisy2);

	// calculate time processing
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
	std::cout << "Reading duration time: " << duration1.count() << " microseconds" << std::endl;

	// start coding CUDA =============================
	
	// allocate memory in CPU for calculation
	float* y_n_real;
	float* y_n_imag; 
	double* X_real; 
	double* X_imag;
	float* R_real; 
	float* R_imag;
	float* Ss_real; 
	float* Ss_imag; 
	float* s_real; 
	float* s_imag;
	float* alpha_real; 

	double* output_real;
	double* output_imag;

	// define our variables
	batch_size = 1; 
	data_num = y_noisy2.n_cols; 
	int y_n_size; 
	int X_size; 
	int R_row; 
	int Ss_size;
	int s_size;
	int alpha_size; 
	int print_flag = 1;						// print report from GPU function
	int num_iter = 1; 


	// prepare our GPU format data
	// convert cx_mat of armadillo library to ordinary arrays. 
	fun1(s, 
		 my_vec, 
		 13, 
		 alpha, 
		 1e-5, 
		 &y_n_real, 
		 &y_n_imag, 
		 &X_real, 
		 &X_imag, 
		 &R_real, 
		 &R_imag,
		 &Ss_real,
		 &Ss_imag,
		 &s_real,
		 &s_imag,
		 &alpha_real,
		 &output_real,
		 &output_imag, 

		 batch_size, 
		 data_num,
		 y_n_size,
		 X_size, 
		 R_row, 
		 Ss_size, 
		 s_size, 
		 alpha_size, 
		 num_iter 
		 );




	// Get the starting timepoint
    auto start22 = std::chrono::high_resolution_clock::now();

	// main GPU kernel
	gpuKernel(y_n_real,
			  y_n_imag,
			  X_real,
			  X_imag,
			  R_real,
			  R_imag,
			  Ss_real,
			  Ss_imag,
			  s_real,
			  s_imag, 
			  alpha_real,
			  output_real,
			  output_imag, 

			  batch_size, 
			  data_num, 
			  y_n_size, 
			  X_size, 
			  R_row, 
			  Ss_size, 
			  s_size, 
			  alpha_size, 
			  num_iter
			  );


	
	// Get the ending timepoint
    auto stop22 = std::chrono::high_resolution_clock::now();


	// calculate time processing
	auto du2 = std::chrono::duration_cast<std::chrono::microseconds>(stop22 - start22);
	std::cout << "GPUUUU Time: " << du2.count() / 1000.0 << " miliseconds" << std::endl;
	std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl; 


	// store result of GPU in an armadillo cx_mat variable
	arma::mat real_part(output_real, X_size, data_num * batch_size); 
	arma::mat imag_part(output_imag, X_size, data_num * batch_size); 
	gpu_apc = arma::cx_mat(real_part, imag_part);
	gpu_apc = gpu_apc.st();


	// free the memory that we use
	free(y_n_real);
	free(y_n_imag); 
	free(X_real); 
	free(X_imag);
	free(R_real); 
	free(R_imag);
	free(Ss_real); 
	free(Ss_imag); 
	free(s_real); 
	free(s_imag); 
	free(alpha_real); 
	free(output_real); 
	free(output_imag); 


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
	clock_t t1 = clock(); 


	// modifying ==========================================
	//while (my_indx < y_noisy2.n_cols) {
	// ====================================================

	std::cout << "indx: " << my_indx << std::endl;
	y_noisy = y_noisy2.col(my_indx); 

                General_APC apc;
                MatchFilter mf;

                cx_mat result_apc = apc.algorithm(s, y_noisy, 13, alpha, 1e-5);
                cx_mat result_mf = mf.algorithm(s, y_noisy, 13);

				for (int k=0; k<batch_size; ++k) {
					double mse = calc_mse(gpu_apc, result_apc.st(), my_indx + k * data_num, 238);

					std::cout << "MSE(" << k << "): " << mse << std::endl;
				}



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




	// modifying ====================================
	//my_indx ++; 
	//}
	// ==============================================
	
	// Get the ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();
	clock_t t2 = clock(); 


	// calculate time processing
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//	std::cout << "CPU: " << duration.count() << " microseconds" << std::endl;
//	printf("CPU: %g\n", (t2-t1)/1000.0); 
	std::cout << "-------------" << std::endl; 
	


}					// void MainWindow::readLine()

