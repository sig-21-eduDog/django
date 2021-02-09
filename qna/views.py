from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from .forms import QuestionForm
from .models import Data
from qna.algorithm.subjects import getSubject


def index(request):
    """View function for home page of site."""
    # post 요청이면 form 데이터 처리
    if request.method == 'POST':
        question_form = QuestionForm(request.POST)
        if question_form.is_valid():
            # 질문이 올바른값이면 여기서 작업처리
            # 지금은 그냥 예시로 질문=Data의 title인 항목 가져옴
            try:
                topic = question_form.clean_questionform()
                content = get_object_or_404(Data, title=topic)
            except:
                content = get_object_or_404(Data, title=getSubject(topic))
                
            # Render the HTML template index.html with the data in the context variable
            context = {
                'content': content,
                'form': question_form,
            }

            return render(request, 'index.html', context=context)

    return render(request, 'index.html', context={'form': QuestionForm, })
