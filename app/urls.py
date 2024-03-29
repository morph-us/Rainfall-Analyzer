from django.urls import path, include
from . import views

urlpatterns = [
    path('index/', views.index, name='index'),
    path('', views.index, name='index'),
    path('region/', views.region, name='region'),
    path('predict/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
]
