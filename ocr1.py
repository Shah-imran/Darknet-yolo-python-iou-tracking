import pymysql
db = pymysql.connect('localhost', 'root', '', 'anpr')
cursor = db.cursor()
mySql_insert_query = """INSERT INTO vehicleLogs (plateNumber, timestamp, imagePath) 
                                VALUES (%s, %s, %s, %s) """
        recordTuple = (name, price, purchase_date)
        cursor.execute(mySql_insert_query, recordTuple)
        connection.commit()
        print("Record inserted successfully into Laptop table")
print(db)