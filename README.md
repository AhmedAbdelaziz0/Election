# Election

Face Recognition part of the system.

1. clone the repository.

2. Install the required packages.

```bash
pip install -r requirements.txt
```

The database should be as follow

```
- database
-- user_name_1
--- image1
--- image2
-- user_name_2
--- image1
--- image2
```

To save new data to the database run `python main.py register`

there must be at least two users.

Run the following command to start the system
```bash
python main.py
```
