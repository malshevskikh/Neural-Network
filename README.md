# Neural-Network
<!-- UNIVERSITY LOGO -->
<div align="center">
  <a href="https://bmstu.ru">
    <img src="https://user-images.githubusercontent.com/67475107/225371733-8fd6f639-bf62-49bd-866c-4e08116fa20c.png" alt="University logo" height="200">
  </a>
  
  Developed by Maxim Alshevskikh (<a href="https://www.linkedin.com/in/maxim-alshevskikh-b473b42b3/">LinkedIn</a>)
  <br/>
</div>

<h2>Лабораторная №1. Кластеризация, снижение размерности пространства признаков.</h2>
<h3>
  Необходимо выбрать некоторый набор данных. Для выбранного набора данных необходимо провести кластеризацию при помощи двух методов, взятых из следующих разных групп: метода k-средних (или альтернативного из группы, основанной на расстояниях между точками), метода DBSCAN (или альтернативного из группы, основанной на плотности точек) или одним из методов иерархической кластеризации. Для метода из первой и третьей групп необходимо выбрать количество кластеров с использованием одного из методов (метод локтя, индекс Дана, индекс Дэвиса-Болдуина, индекс Калинского-Гарабача, метод силуэта, RAND - обязателен при наличии целевой переменной, кофенетическая корреляция). Для методов из второй группы необходимо выбрать ε и другие параметры метода.
</h3>
<h3>
  Требуется сравнить визуализацию данных с использованием методов PCA, MDS (или альтернативного из той же группы, один метод на выбор) с t-SNE, UMAP (или альтернативного из той же группы, один метод на выбор) для случаев кластеризации до и после снижения размерности пространства с разметкой точек по полученным кластерам.
</h3>

<h2>Лабораторная №2. Классификация, оценка точности классификации.</h2>
<h3> Для выбранного набора данных необходимо провести сравнение результатов работы классификаторов из двух групп на выбор: регрессия (линейная, нелинейная, логистическая, метод опорных векторов), k-ближайших соседей;
деревья принятия решений (деревья, случайный лес); бустинг (один из методов на выбор).</h3>
<h3>Для каждого метода необходимо построить матрицу ошибок и рассчитать одну из следующих метрик: f-мера, ROC AUC, accuracy.</h3>
<h3>При обучении классификатора необходимо использовать кроссвалидацию. Необходимо визуализировать изменение точности работы метода на разных шагах кроссвалидации. Необходимо показать как меняется точность классификации при изменении гиперпараметров.</h3>

<h2>Лабораторная №3. Обработка текстов.</h2>
<h3>Необходимо провести векторизацию текстов с использованием двух методов: частота или tf*idf; статические или контекстуализированные векторные модели.</h3>
<h3>Полученные векторы текстов необходимо либо кластеризовать. На полученных кластерах необходимо обучить метод классификации и проверить точность его работы. Необходимо сравнить точность работы метода с и без применения морфологического анализа.</h3>

<h2>Лабораторная №4. Плотные нейронные сети.</h2>
<h3>Необходимо обучить персептрон на одном из использованных выше наборах данных и сравнить точность его работы с использованными ранее методами (см. л/р №2).</h3>
