# Generated by Django 3.1.5 on 2021-01-30 10:18

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.AutoField(db_column='ID', primary_key=True, serialize=False)),
                ('source', models.CharField(db_column='SOURCE', max_length=100)),
                ('cid', models.IntegerField(db_column='CID')),
                ('categoryid', models.IntegerField(db_column='CATEGORYID')),
                ('docid', models.IntegerField(db_column='DOCID')),
                ('title', models.CharField(db_column='TITLE', max_length=150)),
                ('main', models.CharField(db_column='MAIN', max_length=10000)),
            ],
            options={
                'db_table': 'data',
                'managed': False,
            },
        ),
    ]