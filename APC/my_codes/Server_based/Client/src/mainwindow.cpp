#include "../include/mainwindow.h"
#include "ui_mainwindow.h"
#include <QHostAddress>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("Client");


    socket  = new QTcpSocket(this);
    connect(socket,&QTcpSocket::connected,this,&MainWindow::connectedToServer);
    connect(socket,&QTcpSocket::disconnected,this,&MainWindow::disconnected);
    connect(socket,&QTcpSocket::readyRead,this,&MainWindow::readyReadData);
    socket->setReadBufferSize(1e6);

    timer = new QTimer(this);
    connect(timer,&QTimer::timeout,this,&MainWindow::timerTrigged);
    timer->start(1000);
}

MainWindow::~MainWindow()
{
    delete ui;
    socket->close();
    socket->abort();
    timer->stop();
    timer->deleteLater();
    socket->deleteLater();

}

void MainWindow::connectedToServer()
{
    qDebug() << "Connect to Host";
}

void MainWindow::disconnected()
{
    qDebug() << "Disconnected";
}

void MainWindow::readyReadData()
{
    if(socket->bytesAvailable() > 0)
    {
        receiveData.append(socket->readAll());
    }
}

void MainWindow::timerTrigged()
{
    if(socket->state() != QTcpSocket::ConnectedState)
    {
        socket->close();
        socket->abort();
        if(ui->btn_run->isChecked())
            socket->connectToHost(QHostAddress::LocalHost, 50005);
    }
    else
    {
        if(ui->btn_run->isChecked())
        {
            //Send Data
            socket->write((char*)&clientData,sizeof (clientData));

            //Receive Data
            if(receiveData.length() > 0)
            {
                quint16 rHeader = 0;

                QByteArray data  = receiveData.mid(0,sizeof(ServerData));
                ServerData sData = *reinterpret_cast<ServerData*>(data.data());
                receiveData.remove(0,sizeof(ServerData));

                rHeader = sData.header;
                if(rHeader == 0xFA14)
                {
                    ui->lbl_rng->setText(QString::number(sData.range));
                    ui->lbl_az->setText(QString::number(sData.azimuth));
                    ui->lbl_elv->setText(QString::number(sData.elevation));
                }

		// Inserting the main Algorithm ==========
		std::cout << "file_name: " << sData.file_num << std::endl; 
		std::cout << "chunk_num: " << sData.chunk_num << std::endl; 
		std::cout << "step: " << sData.step << std::endl; 

		std::cout << "batch_size: " << sData.batch_size << std::endl; 
        	std::cout << "data_num: " << sData.data_num << std::endl; 
        	std::cout << "offset: " << sData.offset << std::endl; 
        	std::cout << "element_len: " << sData.element_len << std::endl; 
        	std::cout << "out_len: " << sData.out_len << std::endl; 
        	std::cout << "simga: " << sData.sigma << std::endl; 

		// transfer from client to local 
		batch_size = sData.batch_size; 
		data_num = sData.data_num; 
		offset = sData.offset; 
		element_len = sData.element_len; 
		out_len = sData.out_len; 
		sigma = sData.sigma; 

		// store data of the socket to a matrix
		std::vector<arma::cx_mat> my_vec;

		arma::mat real_part(sData.data_I, element_len, data_num); 
		arma::mat imag_part(sData.data_Q, element_len, data_num); 

		y_noisy2 = arma::cx_mat(real_part, imag_part) / std::pow(2.0, 10);

		my_vec.push_back(y_noisy2); 

		// Start coding CUDA --------------------
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

		alpha = arma::mat(2, 1); 
		alpha(0,0) = 1.9;
		alpha(1,0) = 1.8;
//		alpha(2,0) = 1.7; 


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
		data_num = y_noisy2.n_cols; 
		int y_n_size; 
		int X_size; 
		int R_row; 
		int Ss_size;
		int s_size;
		int alpha_size; 
		int num_iter = 1;

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
		     data_num + 2,
		     y_n_size,
		     X_size, 
		     R_row, 
		     Ss_size, 
		     s_size, 
		     alpha_size, 
		     num_iter
		     );
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
			  data_num + 2, 
			  y_n_size, 
			  X_size, 
			  R_row, 
			  Ss_size, 
			  s_size, 
			  alpha_size, 
			  num_iter
			  );
		
		// store result of GPU in an armadillo cx_mat variable
		arma::mat real_section(output_real, X_size, data_num * batch_size); 
		arma::mat imag_section(output_imag, X_size, data_num * batch_size); 
		gpu_apc = arma::cx_mat(real_section, imag_section);
		gpu_apc = gpu_apc.st();

		// =======================================
		while (my_indx < y_noisy2.n_cols) {  
	
			General_APC apc;

			y_noisy = y_noisy2.col(my_indx);
			arma::cx_mat result_apc = apc.algorithm(s, y_noisy, 13, alpha, sigma);

			std::cout << "my_indx: " << my_indx << std::endl; 
			double mse = calc_mse(gpu_apc, result_apc.st(), my_indx , out_len);

			std::cout << "MSE: " << mse << std::endl;
			std::cout << "----------" << std::endl; 

			my_indx ++; // go to the next data row
		}

		my_indx = 0; 
			
		// store result data in the server 
		for (int i = 0; i < data_num; ++i) { 
			for (int j = 0; j < out_len; ++j) {
				clientData.data_I[i * out_len + j] = gpu_apc(i, j).real(); 
				clientData.data_Q[i * out_len + j] = gpu_apc(i, j).imag(); 
			}
		}
		clientData.file_num = sData.file_num; 
		clientData.chunk_num = sData.chunk_num; 
		clientData.step = sData.step; 
		clientData.out_len = sData.out_len; 
		clientData.data_num = sData.data_num; 

		std::cout << "#############################################################################" << std::endl;
		// End of the algorithm ==================

            }
        }
        else
        {
            socket->disconnectFromHost();
            socket->close();
        }

    }
}
