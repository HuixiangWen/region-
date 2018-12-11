from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import os

path = "./newdata"
list1 = os.listdir(path)
print(list1)
list1 = ["0.csv"]
for i in list1:
    print(i)
    Project_data = pd.read_csv(path + "/" + i, encoding="gbk")  # 读取预测数据

    df1 = Project_data[['Rent_amount', 'floor', 'Total_floor', 'Area', 'towards', 'Number_bedroom', 'Number_hall', 'Number_wei', 'location',
                        'subway_route', 'subway_site', 'distance', 'region', 'price']]
    from sklearn.model_selection import train_test_split

    import seaborn as sns

    sns.set()
    m, n = df1.shape
    X = df1.iloc[:, 0: n - 1]
    Y = df1["price"]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    test = pd.concat([x_test, y_test], axis=1)
    # test.to_csv("./t.csv", sep=",", index=0)
    # from sklearn.preprocessing import StandardScaler
    #
    # ss = StandardScaler()
    # x_train = ss.fit_transform(x_train)
    # x_test = ss.transform(x_test)

    regressor = DecisionTreeClassifier(random_state=0)
    parameters = {'max_depth': range(10, 50)}
    scoring_fnc = make_scorer(accuracy_score)
    kfold = KFold(n_splits=10)
    grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold)
    grid = grid.fit(x_train, y_train.ravel())
    reg = grid.best_estimator_
    print('train score: %f' % grid.best_score_)
    print('best parameters:')
    for key in parameters.keys():
        print('%s: %d' % (key, reg.get_params()[key]))
    print('test score: %f' % reg.score(x_test, y_test))

    from sklearn.externals import joblib

    joblib.dump(grid, "./" + i[:-4] + ".m")

    # joblib.dump(grid, "asd1.m")


    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2)
    # clf.fit(x_train, y_train)
    #
    # clf.fit(x_train, y_train.ravel())
    #
    # print(clf.score(x_train, y_train))  # 精度
    # y_hat = clf.predict(x_train)
    # show_accuracy(y_hat, y_train, '训练集')

    # print(clf.score(x_test, y_test))
    # y_hat = clf.predict(x_test)
    # print(list(y_test))
    # print(list(y_hat))




