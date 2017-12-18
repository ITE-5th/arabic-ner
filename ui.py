import json
import re

import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget

from lmbilstmcrf import utils
from lmbilstmcrf.lm_lstm_crf import LM_LSTM_CRF
from lmbilstmcrf.predictor import predict_wc


class Ui_MainWindow(QWidget):
    loc_color = "#FF0000"
    per_color = "#00FF00"
    org_color = "#0000FF"
    misc_color = "#000000"

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(598, 384)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.annotateButton = QtWidgets.QPushButton(self.centralWidget)
        self.annotateButton.setGeometry(QtCore.QRect(250, 310, 89, 25))
        self.annotateButton.setObjectName("annotateButton")
        self.widget = QtWidgets.QWidget(self.centralWidget)
        self.widget.setGeometry(QtCore.QRect(10, 50, 571, 219))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.rawText = QtWidgets.QTextEdit(self.widget)
        self.rawText.setObjectName("rawText")
        self.verticalLayout.addWidget(self.rawText)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.annotatedText = QtWidgets.QTextEdit(self.widget)
        self.annotatedText.setObjectName("annotatedText")
        self.verticalLayout_2.addWidget(self.annotatedText)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # custom
        with open("models/cwlm_lstm_crf.json", 'r') as f:
            jd = json.load(f)
        jd = jd['args']
        checkpoint = torch.load("models/cwlm_lstm_crf.model", map_location=lambda storage, loc: storage)
        f_map = checkpoint['f_map']
        l_map = checkpoint['l_map']
        c_map = checkpoint['c_map']
        in_doc_words = checkpoint['in_doc_words']
        self.model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'],
                                 jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'],
                                 large_CRF=jd['small_crf'], if_highway=jd['high_way'], in_doc_words=in_doc_words,
                                 highway_layers=jd['highway_layers'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model = self.model.cuda()
        self.predictor = predict_wc(True, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'],
                                    l_map['<start>'],
                                    False, 32, jd['caseless'])
        self.setupEvents()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Arabic Ner"))
        self.annotateButton.setText(_translate("MainWindow", "Annotate"))
        self.label.setText(_translate("MainWindow", "Your Raw Text:"))
        self.label_2.setText(_translate("MainWindow", "Annotated:"))

    def setupEvents(self):
        self.annotateButton.clicked.connect(self.predict)

    def predict(self):
        text = self.rawText.toPlainText()
        text = self.prepare_text(text)
        text = text.split("\n")
        features = utils.read_features(text)
        result = self.predictor.output_batch_str(self.model, features)
        result = self.process_result(result)
        print(result)
        self.annotatedText.setText(result)

    def prepare_text(self, text):
        result = ""
        result += "\n".join(text.split(" "))
        return result

    def process_result(self, text):
        temp = r"<span style='color:{};'>\1</span>"
        text = re.sub(r"<LOC>(.+)</LOC>", temp.format(Ui_MainWindow.loc_color), text)
        text = re.sub(r"<PER>(.+)</PER>", temp.format(Ui_MainWindow.per_color), text)
        text = re.sub(r"<ORG>(.+)</ORG>", temp.format(Ui_MainWindow.org_color), text)
        text = re.sub(r"<MISC>(.+)</MISC>", temp.format(Ui_MainWindow.misc_color), text)
        return text
