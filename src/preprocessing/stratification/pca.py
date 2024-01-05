from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA reduced
def get_pca (X_train, X_test, k):
  pca = PCA(n_components=k, random_state = 0)
  pca.fit(X_train)
  X_train_pca = pca.transform(X_train)
  X_test_pca = pca.transform(X_test)

  var_exp = pca.explained_variance_ratio_.cumsum()
  var_exp = var_exp*100
  plt.bar(range(k), var_exp);
  return (X_train_pca, X_test_pca)