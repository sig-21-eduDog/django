from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from .forms import QuestionForm
from .models import Data
from qna.algorithm.subjects import getSubject
from qna.algorithm.documents import search
import usingKorQuAD


def index(request):
    """View function for home page of site."""

    if request.method == 'POST':
        question_form = QuestionForm(request.POST)
        if question_form.is_valid():
            try:
                topic = question_form.clean_questionform()
                try:
                    # Data 테이블에 topic이 있을 때
                    content = get_object_or_404(Data, title=topic)
                except Data.MultipleObjectsReturned:
                    # 여러 개의 본문이 나오면 가장 적합한 본문 하나 선택
                    content = get_object_or_404(Data, main=search(topic))
            except:
                try:
                    # Data 테이블에 topic이 없을 때
                    content = get_object_or_404(Data, title=getSubject(topic))
                except Data.MultipleObjectsReturned:
                    # 여러 개의 본문이 나오면 가장 적합한 본문 하나 선택
                    content = get_object_or_404(Data, main=search(getSubject(topic)))

            # Render the HTML template index.html with the data in the context variable
            context = {
                'content': content,
                'form': question_form,
                'answer': usingKorQuAD.predict_letter(topic, search(getSubject(topic))),
            }

            return render(request, 'index.html', context=context)

    return render(request, 'index.html', context={'form': QuestionForm, })