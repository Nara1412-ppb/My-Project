from django.shortcuts import render,redirect
from django.http import HttpResponse
from .models import destination

# Create your views here.
def index(request):
    dests=destination.objects.all()
    return render(request,'index.html',{'dests':dests})


