from django.shortcuts import render
from .models import product
# Create your views here.
def home(request):
    return render(request, 'index.html')
def index(request):
    products=product.objects.all()
    return render(request, 'index.html', {'products':products})
