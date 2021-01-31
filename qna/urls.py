from django.urls import path
from qna import views


urlpatterns = [
    path('', views.index, name='index'),
]