#include <QApplication>
#include <QWidget>

#include "../include/mainwindow.h"



int main(int argc, char *argv[])
{
    QApplication app(argc, argv);


	MainWindow w;
    w.show();

	/*
    QWidget window;
    window.resize(720, 240);  // Resize the window to 320x240 pixels
    window.setWindowTitle("Simple Example");  // Set the window title
    window.show();  // Show the window
	*/

    return app.exec();  // Start the event loop
}
