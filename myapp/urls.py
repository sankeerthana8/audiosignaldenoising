from django.urls import path
from .views import index

urlpatterns = [
    path('', index, name='index'),
    # Add more paths if needed
]
