import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
@Component({
  selector: 'app-dialog-content',
  templateUrl: 'dialog-content.component.html',
  //styleUrls: ['dialog-content.component.css']
})
export class DialogContentComponent {
  iframeUrl: SafeResourceUrl;
  text:any
  constructor(@Inject(MAT_DIALOG_DATA) public data: { imageSrc: any, html: any, text:any }, private sanitizer: DomSanitizer) {

    const decodedHtml = atob(data.html);

    const dataUrl = 'data:text/html;base64,' + btoa(decodedHtml);

    this.iframeUrl = this.sanitizer.bypassSecurityTrustResourceUrl(dataUrl);
    this.text = data.text
  }
}
