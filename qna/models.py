from django.db import models

# Create your models here.
class Data(models.Model):
    id = models.AutoField(db_column='ID', primary_key=True)  # Field name made lowercase.
    source = models.CharField(db_column='SOURCE', max_length=100)  # Field name made lowercase.
    cid = models.IntegerField(db_column='CID')  # Field name made lowercase.
    categoryid = models.IntegerField(db_column='CATEGORYID')  # Field name made lowercase.
    docid = models.IntegerField(db_column='DOCID')  # Field name made lowercase.
    title = models.CharField(db_column='TITLE', max_length=150)  # Field name made lowercase.
    main = models.CharField(db_column='MAIN', max_length=10000)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'data'
