from PyQt5.QtWidgets import *
from numpy.core.numeric import False_
from unidecode import unidecode  # Para eliminar los acentos

import nltk
# Para dividir el texto en tokens
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords  # Para eliminar las palabras vacías
from nltk.stem import SnowballStemmer  # Para realizar el proceso de Stemming

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split  # Para dividir el dataset
from sklearn.svm import SVC  # Para support vector machine

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt  # Para las gráficas
import pandas as pd  # Para el manejo de datos
import numpy as np  # Par el manejo de arreglos
import re
#import seaborn as sns

from yellowbrick.classifier import ClassificationReport

from gui import UI_principal


class MyApp(QMainWindow):

    def __init__(self):

        super().__init__()
        self.ui = UI_principal.Ui_MainWindow()
        self.ui.setupUi(self)

        self.comentarios = pd.DataFrame()
        self.vector_carateristicas_tf = []
        self.vector_count_vectorizer = []
        self.vector_hashing_vectorizer = []
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        self.ui.pushButton_cargar.clicked.connect(self.buscarArchivo)

        self.ui.pushButton_minuscula.clicked.connect(self.convertirMinuscula)
        self.ui.pushButton_signos_acentos.clicked.connect(self.eliminarAcentos)
        self.ui.pushButton_tokens.clicked.connect(self.tokenizar)
        self.ui.pushButton_palabras_vacias.clicked.connect(
            self.eliminarPalabrasVacias)
        self.ui.pushButton_stemming.clicked.connect(self.stemming)

        self.ui.pushButton_transformar_tf.clicked.connect(
            self.transformarTFIDF)
        self.ui.pushButton_transformar_count.clicked.connect(
            self.transformarCountVectorizer)
        self.ui.pushButton_Hashing.clicked.connect(
            self.transformarHashingVectorizer)

        self.ui.pushButton_dividir_tf.clicked.connect(self.dividirDatasetTFIDF)
        self.ui.pushButton_dividir_count.clicked.connect(
            self.dividirDatasetCountVectorizer)
        self.ui.pushButton_dividir_hashing.clicked.connect(
            self.dividirDatasetHashingVectorize)

        self.ui.pushButton_entrenar_svm.clicked.connect(self.entrenarModelo)

    def buscarArchivo(self):
        # , QDir.homePath(), "All Files (*);;Text Files (*.txt)")
        file, _ = QFileDialog.getOpenFileName(self, 'Buscar Archivo')
        if file:
            #print("Archivo seleccionado: ", file)
            self.comentarios = pd.read_csv(file, sep=";")
            self.agregarContenido('MENSAJES', 0)
            #self.agregarContenido('POLARIDAD', 1)
            self.ui.pushButton_minuscula.setEnabled(True)
            self.ui.pushButton_signos_acentos.setEnabled(True)
            self.ui.pushButton_tokens.setEnabled(True)
            self.ui.pushButton_palabras_vacias.setEnabled(True)
            self.ui.pushButton_stemming.setEnabled(True)
            self.dibujarPastel()

    def dibujarPastel(self):
        labels = list(self.comentarios.POLARIDAD.unique())
        valores = dict(self.comentarios.POLARIDAD.value_counts())
        explode = (0.2, 0, 0)
        # print(dict(values))
        self.ui.MplWidget.canvas.total_data.clear()
        self.ui.MplWidget.canvas.total_data.pie(valores.values(), labels=valores.keys(), autopct='%1.1f%%',
                                                shadow=True, startangle=90, textprops={'size': 'x-small'})
        self.ui.MplWidget.canvas.total_data.axis('equal')
        self.ui.MplWidget.canvas.draw()

    def agregarContenido(self, key, numColum):
        fila = 0
        self.ui.tableWidget_datos.setRowCount(len(self.comentarios.index))
        for registro in self.comentarios[key]:
            columna = numColum
            celda = QTableWidgetItem(str(registro))
            self.ui.tableWidget_datos.setItem(fila, columna, celda)
            fila += 1

    def convertirMinuscula(self):
        try:
            self.comentarios['minusculas'] = self.comentarios['MENSAJES'].str.lower(
            )
            self.agregarContenido('minusculas', 1)
        except:
            QMessageBox.warning(self, "Advertencia",
                                "No existe ningún archivo para procesar")

    def eliminarAcentos(self):
        try:
            reg_signos = re.compile('[^\w\s]')
            self.comentarios['puntuacion'] = self.comentarios['minusculas'].replace(
                reg_signos, '')
            self.comentarios['acentos'] = self.comentarios['puntuacion'].apply(
                unidecode)
            self.agregarContenido('acentos', 2)
        except:
            QMessageBox.warning(self, "Advertencia",
                                "Por favor, Realice el paso anterior")

    def tokenizar(self):
        try:
            self.comentarios['tokens'] = self.comentarios['acentos'].apply(
                word_tokenize)
            self.agregarContenido('tokens', 3)
        except:
            QMessageBox.warning(self, "Advertencia",
                                "Por favor, Realice el paso anterior")

    def eliminarPalabrasVacias(self):
        try:
            stop_words = set(stopwords.words('spanish'))
            self.comentarios['stopwords'] = self.comentarios['tokens'].apply(
                lambda msg: [item for item in msg if item not in stop_words])
            self.agregarContenido('stopwords', 4)
        except:
            QMessageBox.warning(self, "Advertencia",
                                "Por favor, Realice el paso anterior")

    def stemming(self):
        try:
            stemmer = SnowballStemmer('spanish')
            self.comentarios['stemming'] = self.comentarios['stopwords'].apply(
                lambda msg: [stemmer.stem(item) for item in msg])
            self.agregarContenido('stemming', 5)
        except:
            QMessageBox.warning(self, "Advertencia",
                                "Por favor, Realice el paso anterior")

    def transformarTFIDF(self):
        try:
            encoding = self.ui.lineEdit_encoding_tf.text()
            decode_error = self.ui.lineEdit_decode_error_tf.text()
            strip_accents = self.ui.lineEdit_strip_accents_tf.text()
            lowercase = self.ui.lineEdit_lowercase_tf.text()
            preprocessor = self.ui.lineEdit_preprocessor_tf.text()
            tokenizer = self.ui.lineEdit_tokenizer_tf.text()
            stop_words = self.ui.lineEdit_stop_words_tf.text()
            token_pattern = self.ui.lineEdit_token_pattern_tf.text()
            ngram_range = tuple(
                map(int, self.ui.lineEdit_ngram_tf.text().split(',')))
            analyzer = self.ui.lineEdit_analyzer_tf.text()
            max_df = float(self.ui.lineEdit_max_df_tf.text())
            min_df = self.ui.lineEdit_min_df_tf.text()
            max_features = self.ui.lineEdit_max_features_tf.text()
            vocabulary = self.ui.lineEdit_vocabulary_tf.text()
            binary = self.ui.lineEdit_binary_tf.text()
            dtype = self.ui.lineEdit_dtype_tf.text()
            norm = self.ui.lineEdit_norm_tf.text()
            use_idf = self.ui.lineEdit_use_idf_tf.text()
            smooth_idf = self.ui.lineEdit_smooth_idf_tf.text()
            sublinear_tf = self.ui.lineEdit_sublinear_tf.text()

            if strip_accents == 'None':
                strip_accents = None

            if lowercase == 'True':
                lowercase = True
            elif lowercase == 'False':
                lowercase = False

            if preprocessor == 'None':
                preprocessor = None

            if tokenizer == 'None':
                tokenizer = None

            if stop_words == 'None':
                stop_words = None

            if min_df == '0' or min_df == '1':
                min_df = int(min_df)
            else:
                min_df = float(min_df)

            if max_features == 'None':
                max_features = None
            else:
                max_features = int(max_features)

            if vocabulary == 'None':
                vocabulary = None

            if binary == 'True':
                binary = True
            elif binary == 'False':
                binary = False

            if dtype == 'float64':
                dtype = np.float64

            if use_idf == 'True':
                use_idf = True
            elif use_idf == 'False':
                use_idf = False

            if smooth_idf == 'True':
                smooth_idf = True
            elif smooth_idf == 'False':
                smooth_idf = False

            if sublinear_tf == 'True':
                sublinear_tf = True
            elif sublinear_tf == 'False':
                sublinear_tf = False

            vectorizer_tf = TfidfVectorizer(
                encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

            self.vector_carateristicas_tf = vectorizer_tf.fit_transform(
                self.comentarios['stemming'].agg(' '.join))

            QMessageBox.information(self, "Información",
                                    "Documento Transformado correctamente con TF-IDF")

        except ValueError as err:
            QMessageBox.warning(self, "Advertencia",
                                "ValueError: {0}".format(err))

    def transformarCountVectorizer(self):
        try:
            encoding = self.ui.lineEdit_encoding_count.text()
            decode_error = self.ui.lineEdit_decode_error_count.text()
            strip_accents = self.ui.lineEdit_strip_accents_count.text()
            lowercase = self.ui.lineEdit_lowercase_count.text()
            preprocessor = self.ui.lineEdit_preprocessor_count.text()
            tokenizer = self.ui.lineEdit_tokenizer_count.text()
            stop_words = self.ui.lineEdit_stop_words_count.text()
            token_pattern = self.ui.lineEdit_token_pattern_count.text()
            ngram_range = tuple(
                map(int, self.ui.lineEdit_ngram_count.text().split(',')))
            analyzer = self.ui.lineEdit_analyzer_count.text()
            max_df = float(self.ui.lineEdit_max_df_count.text())
            min_df = self.ui.lineEdit_min_df_count.text()
            max_features = self.ui.lineEdit_max_features_count.text()
            vocabulary = self.ui.lineEdit_vocabulary_count.text()
            binary = self.ui.lineEdit_binary_count.text()
            dtype = self.ui.lineEdit_dtype_count.text()

            if strip_accents == 'None':
                strip_accents = None

            if lowercase == 'True':
                lowercase = True
            elif lowercase == 'False':
                lowercase = False

            if preprocessor == 'None':
                preprocessor = None

            if tokenizer == 'None':
                tokenizer = None

            if stop_words == 'None':
                stop_words = None

            if min_df == '0' or min_df == '1':
                min_df = int(min_df)
            else:
                min_df = float(min_df)

            if max_features == 'None':
                max_features = None
            else:
                max_features = int(max_features)

            if vocabulary == 'None':
                vocabulary = None

            if binary == 'True':
                binary = True
            elif binary == 'False':
                binary = False

            if dtype == 'int64':
                dtype = np.int64

            vectorizer_count = CountVectorizer(
                encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)

            self.vector_count_vectorizer = vectorizer_count.fit_transform(
                self.comentarios['stemming'].agg(' '.join))

            QMessageBox.information(self, "Información",
                                    "Documento Transformado correctamente con CountVectorizer")

        except ValueError as err:
            QMessageBox.warning(self, "Advertencia",
                                "ValueError: {0}".format(err))

    def transformarHashingVectorizer(self):
        try:
            encoding = self.ui.lineEdit_encoding_hashing.text()
            decode_error = self.ui.lineEdit_decode_error_hashing.text()
            strip_accents = self.ui.lineEdit_strip_accents_hashing.text()
            lowercase = self.ui.lineEdit_lowercase_hashing.text()
            preprocessor = self.ui.lineEdit_preprocessor_hashing.text()
            tokenizer = self.ui.lineEdit_tokenizer_hashing.text()
            stop_words = self.ui.lineEdit_stop_words_hashing.text()
            token_pattern = self.ui.lineEdit_token_pattern_hashing.text()
            ngram_range = tuple(
                map(int, self.ui.lineEdit_ngram_hashing.text().split(',')))
            analyzer = self.ui.lineEdit_analyzer_hashing.text()
            n_features = int(self.ui.lineEdit_n_features_hashing.text())
            binary = self.ui.lineEdit_binary_hashing.text()
            norm = self.ui.lineEdit_norm_hashing.text()
            alternate_sign = self.ui.lineEdit_alternate_sign_hashing.text()
            dtype = self.ui.lineEdit_dtype_hashing.text()

            if strip_accents == 'None':
                strip_accents = None

            if lowercase == 'True':
                lowercase = True
            elif lowercase == 'False':
                lowercase = False

            if preprocessor == 'None':
                preprocessor = None

            if tokenizer == 'None':
                tokenizer = None

            if stop_words == 'None':
                stop_words = None

            if binary == 'True':
                binary = True
            elif binary == 'False':
                binary = False

            if alternate_sign == 'True':
                alternate_sign = True
            elif alternate_sign == 'False':
                alternate_sign = False

            if dtype == 'float64':
                dtype = np.float64

            vectorizer_hashing = HashingVectorizer(
                encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, n_features=n_features, binary=binary, norm=norm, alternate_sign=alternate_sign, dtype=dtype)

            self.vector_hashing_vectorizer = vectorizer_hashing.fit_transform(
                self.comentarios['stemming'].agg(' '.join))

            QMessageBox.information(self, "Información",
                                    "Documento Transformado correctamente con HashingVectorizer")

        except ValueError as err:
            QMessageBox.warning(self, "Advertencia",
                                "ValueError: {0}".format(err))

    def dividirDatasetTFIDF(self):
        self.trainTestSplit(self.vector_carateristicas_tf)

    def dividirDatasetCountVectorizer(self):
        self.trainTestSplit(self.vector_count_vectorizer)

    def dividirDatasetHashingVectorize(self):
        self.trainTestSplit(self.vector_hashing_vectorizer)

    def trainTestSplit(self, vector):
        try:
            test_size = self.ui.lineEdit_test_size.text()
            train_size = self.ui.lineEdit_train_size.text()
            random_state = self.ui.lineEdit_random_state.text()
            shuffle = self.ui.lineEdit_shuffle.text()
            stratify = self.ui.lineEdit_stratify.text()

            if test_size == 'None':
                test_size = None
            else:
                test_size = float(test_size)

            if train_size == 'None':
                train_size = None
            else:
                train_size = float(train_size)

            if random_state == 'None':
                random_state == None
            else:
                random_state = int(random_state)

            if shuffle == 'True':
                shuffle = True
            elif shuffle == 'False':
                shuffle = False

            if stratify == 'None':
                stratify = None
            else:
                stratify = np.empty_like(stratify)

            labels = self.comentarios['POLARIDAD'].iloc[:].values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                vector, labels, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)

        except ValueError as err:
            QMessageBox.warning(self, "Advertencia",
                                "ValueError: {0}".format(err))
        #self.ui.label_dividir.setText(f'A seleccionado {test_size*100}% de los datos para el entrenamiento')
        # self.ui.label_vector.setWordWrap(True)

    def entrenarModelo(self):
        C = float(self.ui.lineEdit_c.text())
        kernel = self.ui.lineEdit_kernel.text()
        degree = int(self.ui.lineEdit_degree.text())
        gamma = self.ui.lineEdit_gamma.text()
        coef0 = float(self.ui.lineEdit_coef.text())
        shrinking = self.ui.lineEdit_shrinkg.text()
        probability = self.ui.lineEdit_probability.text()
        tol = float(self.ui.lineEdit_tol.text())
        cache_size = float(self.ui.lineEdit_cache.text())
        class_weight = self.ui.lineEdit_class.text()
        verbose = self.ui.lineEdit_verbose.text()
        max_iter = int(self.ui.lineEdit_max.text())
        decision_function_shape = self.ui.lineEdit_decision.text()
        break_ties = self.ui.lineEdit_break.text()
        random_state = self.ui.lineEdit_random.text()

        if shrinking == 'True':
            shrinking = True
        elif shrinking == 'False':
            shrinking = False

        if probability == 'True':
            probability = True
        elif probability == 'False':
            probability = False

        if class_weight == 'None':
            class_weight = None

        if verbose == 'True':
            verbose = True
        elif verbose == 'False':
            verbose = False

        if break_ties == 'True':
            break_ties = True
        elif break_ties == 'False':
            break_ties = False

        if random_state == 'None':
            random_state = None
        else:
            random_state = int(random_state)

        clasificador_SVM = SVC(C=C, kernel=kernel, degree=degree,
                               gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)

        clasificador_SVM.fit(self.X_train, self.y_train)
        predictions_svm = clasificador_SVM.predict(self.X_test)

        reporte = classification_report(self.y_test, predictions_svm)
        self.ui.label_resultado_svm.setText(reporte)

        #self.ui.label_resultado_svm.setWordWrap(True)

        visualizer = ClassificationReport(clasificador_SVM, support=True)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

        # if clasificador_SVM:
        #    QMessageBox.warning(self, "Advertencia",
        #        "Crado correctamente")
        #    clasificador_SVM.fit(self.X_train, self.y_train)
        #    print(clasificador_SVM)
        
