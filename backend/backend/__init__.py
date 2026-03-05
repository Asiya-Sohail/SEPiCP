import pymysql

# Spoof the version to satisfy Django's requirement for mysqlclient >= 2.2.1
pymysql.version_info = (2, 2, 1, 'final', 0)
pymysql.install_as_MySQLdb()
