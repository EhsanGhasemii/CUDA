/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label_7;
    QLabel *label_5;
    QLabel *label_14;
    QLabel *label_4;
    QLabel *label_6;
    QLabel *label_15;
    QLabel *lbl_rng;
    QLabel *lbl_az;
    QLabel *lbl_elv;
    QPushButton *btn_run;
    QHBoxLayout *horizontalLayout;
    QLabel *lbl_alpha;
    QComboBox *comboBox;
    QPushButton *submitButton;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(325, 200);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout_2 = new QGridLayout(centralwidget);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_7, 5, 0, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy);
        label_5->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_5, 2, 1, 1, 1);

        label_14 = new QLabel(groupBox);
        label_14->setObjectName(QString::fromUtf8("label_14"));
        sizePolicy.setHeightForWidth(label_14->sizePolicy().hasHeightForWidth());
        label_14->setSizePolicy(sizePolicy);
        label_14->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_14, 3, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_4, 2, 0, 1, 1);

        label_6 = new QLabel(groupBox);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout->addWidget(label_6, 3, 0, 2, 1);

        label_15 = new QLabel(groupBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));
        sizePolicy.setHeightForWidth(label_15->sizePolicy().hasHeightForWidth());
        label_15->setSizePolicy(sizePolicy);
        label_15->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_15, 5, 1, 1, 1);

        lbl_rng = new QLabel(groupBox);
        lbl_rng->setObjectName(QString::fromUtf8("lbl_rng"));
        lbl_rng->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(lbl_rng, 2, 2, 1, 1);

        lbl_az = new QLabel(groupBox);
        lbl_az->setObjectName(QString::fromUtf8("lbl_az"));
        lbl_az->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(lbl_az, 3, 2, 1, 1);

        lbl_elv = new QLabel(groupBox);
        lbl_elv->setObjectName(QString::fromUtf8("lbl_elv"));
        lbl_elv->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(lbl_elv, 5, 2, 1, 1);


        gridLayout_2->addWidget(groupBox, 0, 0, 1, 1);

        btn_run = new QPushButton(centralwidget);
        btn_run->setObjectName(QString::fromUtf8("btn_run"));
        btn_run->setCheckable(true);

        gridLayout_2->addWidget(btn_run, 1, 0, 1, 1);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        lbl_alpha = new QLabel(centralwidget);
        lbl_alpha->setObjectName(QString::fromUtf8("lbl_alpha"));

        horizontalLayout->addWidget(lbl_alpha);

        comboBox = new QComboBox(centralwidget);
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->setObjectName(QString::fromUtf8("comboBox"));

        horizontalLayout->addWidget(comboBox);

        submitButton = new QPushButton(centralwidget);
        submitButton->setObjectName(QString::fromUtf8("submitButton"));

        horizontalLayout->addWidget(submitButton);


        gridLayout_2->addLayout(horizontalLayout, 2, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Server Data", nullptr));
        label_7->setText(QCoreApplication::translate("MainWindow", "Elevation", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", ":", nullptr));
        label_14->setText(QCoreApplication::translate("MainWindow", ":", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Range", nullptr));
        label_6->setText(QCoreApplication::translate("MainWindow", "Azimuth", nullptr));
        label_15->setText(QCoreApplication::translate("MainWindow", ":", nullptr));
        lbl_rng->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        lbl_az->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        lbl_elv->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        btn_run->setText(QCoreApplication::translate("MainWindow", "Run", nullptr));
        lbl_alpha->setText(QCoreApplication::translate("MainWindow", "Select Alpha:", nullptr));
        comboBox->setItemText(0, QCoreApplication::translate("MainWindow", "1", nullptr));
        comboBox->setItemText(1, QCoreApplication::translate("MainWindow", "2", nullptr));
        comboBox->setItemText(2, QCoreApplication::translate("MainWindow", "3", nullptr));

        submitButton->setText(QCoreApplication::translate("MainWindow", "Submit", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
