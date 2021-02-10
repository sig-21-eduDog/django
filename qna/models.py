from django.db import models
from django.urls import reverse
from qna.algorithm.documents import search


# 데이터 테이블
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

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        """Returns the url to access a particular instance of the model."""
        return reverse('model-detail-view', args=[str(self.id)])
