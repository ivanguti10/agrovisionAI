import { ComponentFixture, TestBed } from '@angular/core/testing';

import { XaiComponent } from './xai.component';

describe('XaiComponent', () => {
  let component: XaiComponent;
  let fixture: ComponentFixture<XaiComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [XaiComponent]
    });
    fixture = TestBed.createComponent(XaiComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
