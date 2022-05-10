import sqlite3


conn = sqlite3.connect('history_database.db')
conn.execute("PRAGMA foreign_keys = 1")
cursor = conn.cursor()


class DatabaseInput:

    def __init__(self):
        self.dbo = DatabaseOutput()

    def tables_check(self):
        try:
            cursor.execute("SELECT * FROM ImageData")
            return True
        except sqlite3.OperationalError:
            return False

    # Создание таблицы данных изображения
    def create_tables(self):
        cursor.execute('''CREATE TABLE IF NOT EXISTS ImageData
        (image_name TEXT NOT NULL PRIMARY KEY,
        image_array BLOB NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS ImageCaps
        (image_id INTEGER NOT NULL,
        image_name TEXT NOT NULL,
        caption TEXT NOT NULL,
        PRIMARY KEY (image_id, image_name) )''')

    # Подготовка данных для таблица ImageData
    def image_data(self, filename1, filename2):
        f = open(filename2, 'rb')
        image_array = f.read()
        f.close()
        # print(f'{sys.getsizeof(image_array) / (8 * 1024)} Кбайт')
        # print((filename1, image_array))
        # print(len(filename1,), len(image_array), len((filename1, image_array)))
        return filename1, image_array

    # Подготовка данных таблицы ImageCaps
    def image_caps(self, filename1, caption):
        image_id = len(self.dbo.image_id_from_db())
        # print((image_id, filename1, caption))
        return image_id, filename1, caption,

    # Заполнение данными таблицы
    def fill_tables(self, filename1, filename2, caption):
        self.create_tables()
        self.image_data(filename1, filename2)
        self.image_caps(filename1, caption)
        try:
            cursor.execute("INSERT INTO ImageData VALUES (?,?)", self.image_data(filename1, filename2))
            cursor.execute("INSERT INTO ImageCaps VALUES (?,?,?)", self.image_caps(filename1, caption))
        except sqlite3.IntegrityError:
            pass

        conn.commit()


class DatabaseOutput:

    def __init__(self):
        pass

    # Вывод информации из базы данных в нужном формате
    def image_array_from_db(self):
        query = "SELECT image_array from ImageData"
        cursor.execute(query)
        rows = cursor.fetchall()
        rowslist = []
        for i in range(len(rows)):
            rowslist.append(rows[i][0])
        return rowslist

    def image_id_from_db(self):
        query = "SELECT image_id from ImageCaps"
        cursor.execute(query)
        nums = cursor.fetchall()
        numlist = []
        for i in range(len(nums)):
            numlist.append(nums[i][0])
        return numlist

    def image_caption_from_db(self):
        query = "SELECT caption from ImageCaps"
        cursor.execute(query)
        rows = cursor.fetchall()
        # print(rows)
        rowslist = []
        for i in range(len(rows)):
            rowslist.append(rows[i][0])
        # print(rowslist)
        return rowslist

    def drop_tables(self):
        cursor.execute('DROP TABLE ImageData')
        cursor.execute('DROP TABLE ImageCaps')


