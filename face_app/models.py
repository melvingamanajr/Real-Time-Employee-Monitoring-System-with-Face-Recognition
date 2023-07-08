from django.db import models


# Set up the Django model for storing employee data
class Employee(models.Model):
    name = models.CharField(max_length=100)
    image = models.BinaryField()
    location = models.CharField(max_length=100)
    datetime = models.DateField(auto_now_add=True)