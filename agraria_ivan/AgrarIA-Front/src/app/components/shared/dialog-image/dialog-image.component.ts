import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';

@Component({
  selector: 'app-dialog-image',
  templateUrl: './dialog-image.component.html',
  styleUrls: ['./dialog-image.component.scss']
})
export class DialogImageComponent {
  text:any
  constructor(@Inject(MAT_DIALOG_DATA) public data: { imageSrc: any, text:any, imagenNormal:any , imageCombined:any }) {
    this.text = data.text
   }
}

