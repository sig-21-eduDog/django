from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy

class QuestionForm(forms.Form):
    question = forms.CharField(label='', widget=forms.TextInput(attrs={'placeholder': '질문을 입력해주세요.', 'class': 'form-control'}))

    def clean_questionform(self):
        data = self.cleaned_data['question']
        if data == '':
            raise ValidationError(ugettext_lazy('잘못된 문자 - 공백'))
        return data
